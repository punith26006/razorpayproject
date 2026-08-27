import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';
import { ReturnsModule } from './returns/returns.module';
import { DisputesModule } from './disputes/disputes.module';
import { MetricsModule } from './metrics/metrics.module';

@Module({
  imports: [
    ConfigModule.forRoot({ isGlobal: true }),
    ReturnsModule,
    DisputesModule,
    MetricsModule,
  ],
})
export class AppModule {}
