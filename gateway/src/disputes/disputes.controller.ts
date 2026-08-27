import { Body, Controller, HttpCode, HttpStatus, Post } from '@nestjs/common';
import { DisputesService } from './disputes.service';
import { CreateDisputeEvidenceDto } from './dto/create-dispute.dto';

@Controller('api/v1/disputes')
export class DisputesController {
  constructor(private readonly disputesService: DisputesService) {}

  @Post('evidence')
  @HttpCode(HttpStatus.OK)
  async generateEvidence(@Body() dto: CreateDisputeEvidenceDto) {
    return this.disputesService.generateEvidencePackage(dto);
  }

  @Post('webhook')
  @HttpCode(HttpStatus.OK)
  async handleWebhook(@Body() webhookPayload: any) {
    return this.disputesService.handleRazorpayWebhook(webhookPayload);
  }
}
