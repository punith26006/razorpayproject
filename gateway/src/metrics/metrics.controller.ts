import { Controller, Get } from '@nestjs/common';
import { MetricsService } from './metrics.service';

@Controller('api/v1/metrics')
export class MetricsController {
  constructor(private readonly metricsService: MetricsService) {}

  @Get('performance')
  getPerformance() {
    return this.metricsService.getBenchmarkMetrics();
  }

  @Get('cost-curve')
  getCostCurve() {
    return this.metricsService.getCostCurveData();
  }
}
