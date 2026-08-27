import { IsArray, IsNotEmpty, IsNumber, IsOptional, IsString } from 'class-validator';

export class TransactionDetailDto {
  @IsString()
  @IsNotEmpty()
  transactionId: string;

  @IsNumber()
  amount: number;

  @IsString()
  @IsOptional()
  currency?: string;

  @IsString()
  @IsOptional()
  productCategory?: string;

  @IsString()
  @IsOptional()
  shippingAddress?: string;

  @IsString()
  @IsOptional()
  billingAddress?: string;
}

export class CreateDisputeEvidenceDto {
  @IsNotEmpty()
  transaction: TransactionDetailDto;

  @IsArray()
  @IsOptional()
  customerHistory?: any[];

  @IsOptional()
  includeRiskScore?: boolean;
}
