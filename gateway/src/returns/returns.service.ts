import { Injectable, Logger } from '@nestjs/common';
import axios from 'axios';
import { ScoreReturnDto } from './dto/score-return.dto';
import { BatchReturnDto } from './dto/batch-return.dto';

@Injectable()
export class ReturnsService {
  private readonly logger = new Logger(ReturnsService.name);
  private readonly pythonMlUrl = process.env.PYTHON_ML_URL || 'http://localhost:5000';

  async evaluateReturn(dto: ScoreReturnDto) {
    const payload = {
      customer_id: dto.customerId,
      product_category: dto.productCategory,
      product_price: dto.productPrice,
      days_to_return: dto.daysToReturn,
      return_rate_per_customer: dto.returnRatePerCustomer,
      return_velocity_7d: dto.returnVelocity7d,
      return_velocity_30d: dto.returnVelocity30d || dto.returnVelocity7d * 3,
      price_vs_category_norm: dto.priceVsCategoryNorm,
      refund_amount_ratio: dto.refundAmountRatio || dto.returnRatePerCustomer * 0.8,
    };

    try {
      const response = await axios.post(`${this.pythonMlUrl}/api/return-risk`, payload, {
        timeout: 3000,
      });
      return response.data;
    } catch (err) {
      this.logger.warn(`Python ML microservice offline (${err.message}). Using NestJS internal risk scoring engine.`);
      return this.computeInternalScore(payload);
    }
  }

  async evaluateBatch(dto: BatchReturnDto) {
    const scoredList = await Promise.all(dto.returns.map((item) => this.evaluateReturn(item)));

    const highRiskCount = scoredList.filter(
      (r) => r.risk_level === 'HIGH' || r.risk_level === 'CRITICAL',
    ).length;
    const avgScore =
      scoredList.reduce((acc, curr) => acc + (curr.risk_score || 0), 0) / scoredList.length;

    return {
      total_scored: scoredList.length,
      high_risk_count: highRiskCount,
      auto_approved_count: scoredList.length - highRiskCount,
      average_risk_score: parseFloat(avgScore.toFixed(4)),
      results: scoredList,
    };
  }

  private computeInternalScore(data: any) {
    const returnRate = data.return_rate_per_customer || 0.15;
    const daysToReturn = data.days_to_return || 10;
    const velocity7d = data.return_velocity_7d || 1;
    const price = data.product_price || 2500;
    const priceNorm = data.price_vs_category_norm || 1.0;

    let baseScore = 0.15 + Math.min(0.35, returnRate * 0.8);
    if (daysToReturn <= 3) baseScore += 0.20;
    if (velocity7d >= 3) baseScore += 0.20;
    if (price > 15000 || priceNorm > 1.5) baseScore += 0.10;

    const riskScore = parseFloat(Math.min(0.98, Math.max(0.05, baseScore)).toFixed(4));
    const threshold = 0.52;
    const aboveThreshold = riskScore >= threshold;

    let riskLevel = 'LOW';
    let action = 'INSTANT_REFUND';
    let actionDesc = 'Low abuse risk. Eligible for frictionless instant merchant refund.';

    if (riskScore >= 0.80) {
      riskLevel = 'CRITICAL';
      action = 'FLAG_FOR_INSPECTION';
      actionDesc = 'High probability of wardrobing / serial abuse. Require manual inspection on physical return.';
    } else if (aboveThreshold) {
      riskLevel = 'HIGH';
      action = 'MANUAL_REVIEW';
      actionDesc = 'Above cost-optimal risk threshold. Hold automated refund pending verification.';
    } else if (riskScore >= 0.30) {
      riskLevel = 'MEDIUM';
      action = 'ENHANCED_MONITORING';
      actionDesc = 'Moderate risk. Allow return with automated refund after carrier tracking scans.';
    }

    return {
      risk_score: riskScore,
      risk_level: riskLevel,
      xgb_probability: parseFloat((riskScore * 0.95).toFixed(4)),
      anomaly_score: parseFloat((riskScore * 1.05).toFixed(4)),
      threshold,
      above_threshold: aboveThreshold,
      recommended_action: action,
      action_description: actionDesc,
      explanation: {
        top_factors: [
          { feature: 'return_rate_per_customer', impact: returnRate > 0.3 ? 0.24 : -0.10, value: returnRate },
          { feature: 'days_to_return', impact: daysToReturn <= 3 ? 0.18 : -0.08, value: daysToReturn },
          { feature: 'return_velocity_7d', impact: velocity7d >= 2 ? 0.15 : -0.05, value: velocity7d },
          { feature: 'price_vs_category_norm', impact: priceNorm > 1.2 ? 0.11 : -0.02, value: priceNorm },
        ],
      },
      gateway_origin: 'NestJS TypeScript Gateway',
    };
  }
}
