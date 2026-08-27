import { Body, Controller, HttpCode, HttpStatus, Post } from '@nestjs/common';
import { ReturnsService } from './returns.service';
import { ScoreReturnDto } from './dto/score-return.dto';
import { BatchReturnDto } from './dto/batch-return.dto';

@Controller('api/v1/returns')
export class ReturnsController {
  constructor(private readonly returnsService: ReturnsService) {}

  @Post('score')
  @HttpCode(HttpStatus.OK)
  async scoreReturn(@Body() dto: ScoreReturnDto) {
    return this.returnsService.evaluateReturn(dto);
  }

  @Post('batch')
  @HttpCode(HttpStatus.OK)
  async scoreBatch(@Body() dto: BatchReturnDto) {
    return this.returnsService.evaluateBatch(dto);
  }
}
