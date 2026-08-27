import { NestFactory } from '@nestjs/core';
import { Logger, ValidationPipe } from '@nestjs/common';
import { AppModule } from './app.module';

async function bootstrap() {
  const logger = new Logger('AI-Risk-Manager-Gateway');
  const app = await NestFactory.create(AppModule);

  app.enableCors({
    origin: '*',
    methods: 'GET,HEAD,PUT,PATCH,POST,DELETE',
  });

  app.useGlobalPipes(
    new ValidationPipe({
      whitelist: true,
      transform: true,
      forbidNonWhitelisted: false,
    }),
  );

  const port = process.env.PORT || 3000;
  await app.listen(port);
  logger.log(`🛡️ NestJS Merchant Risk Gateway is running on: http://localhost:${port}`);
  logger.log(`Endpoints available:`);
  logger.log(`  -> POST http://localhost:${port}/api/v1/returns/score`);
  logger.log(`  -> POST http://localhost:${port}/api/v1/returns/batch`);
  logger.log(`  -> POST http://localhost:${port}/api/v1/disputes/evidence`);
  logger.log(`  -> POST http://localhost:${port}/api/v1/disputes/webhook (Razorpay Webhook)`);
  logger.log(`  -> GET  http://localhost:${port}/api/v1/metrics/cost-curve`);
  logger.log(`  -> GET  http://localhost:${port}/api/v1/metrics/performance`);
}

bootstrap();
