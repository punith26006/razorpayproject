import { IsNotEmpty, IsNumber, IsOptional, IsString, Max, Min } from 'class-validator';

export class ScoreReturnDto {
  @IsString()
  @IsNotEmpty()
  customerId: string;

  @IsString()
  @IsNotEmpty()
  productCategory: string;

  @IsNumber()
  @Min(1)
  productPrice: number;

  @IsNumber()
  @Min(0)
  daysToReturn: number;

  @IsNumber()
  @Min(0)
  @Max(1)
  returnRatePerCustomer: number;

  @IsNumber()
  @Min(0)
  returnVelocity7d: number;

  @IsNumber()
  @Min(0)
  @IsOptional()
  returnVelocity30d?: number;

  @IsNumber()
  @Min(0)
  priceVsCategoryNorm: number;

  @IsNumber()
  @Min(0)
  @Max(1)
  @IsOptional()
  refundAmountRatio?: number;
}
