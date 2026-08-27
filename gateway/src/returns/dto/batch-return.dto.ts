import { Type } from 'class-transformer';
import { ArrayMinSize, IsArray, ValidateNested } from 'class-validator';
import { ScoreReturnDto } from './score-return.dto';

export class BatchReturnDto {
  @IsArray()
  @ArrayMinSize(1)
  @ValidateNested({ each: true })
  @Type(() => ScoreReturnDto)
  returns: ScoreReturnDto[];
}
