#!/usr/bin/env bash
# Creates the public upload API behind the dashboard. Run once from anywhere: bash winnow/api/deploy.sh
set -euo pipefail

ACCOUNT=073868306412
BUCKET=winnow-test-$ACCOUNT
SITE_ORIGIN=https://d1oc3ay9n04ubj.cloudfront.net

cd "$(dirname "$0")"
python -c "import zipfile; z = zipfile.ZipFile('winnow-api.zip', 'w', zipfile.ZIP_DEFLATED); z.write('app.py'); z.close()"

aws lambda create-function \
  --function-name winnow-api \
  --runtime python3.12 \
  --architectures arm64 \
  --handler app.handler \
  --timeout 15 \
  --role "arn:aws:iam::$ACCOUNT:role/winnowApiRole" \
  --zip-file fileb://winnow-api.zip \
  --environment "Variables={WINNOW_BUCKET=$BUCKET,WINNOW_CLUSTER=winnow-cluster,WINNOW_TASK_DEF=winnow-task,WINNOW_SUBNET=subnet-0387269f363b2765f,WINNOW_SECURITY_GROUP=sg-0f11396bcb184f8c1}" \
  --query FunctionArn --output text
aws lambda wait function-active-v2 --function-name winnow-api

aws lambda create-function-url-config \
  --function-name winnow-api \
  --auth-type NONE \
  --cors "AllowOrigins=$SITE_ORIGIN,AllowMethods=POST,AllowHeaders=content-type,MaxAge=300" \
  --query FunctionUrl --output text

# Public function URLs need both statements: one for the URL, one for the invoke it performs.
aws lambda add-permission --function-name winnow-api --statement-id public-url \
  --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE > /dev/null
aws lambda add-permission --function-name winnow-api --statement-id public-url-invoke \
  --action lambda:InvokeFunction --principal "*" --invoked-via-function-url > /dev/null

# Browsers upload straight to S3 from the dashboard, so S3 must accept POSTs from that one site.
aws s3api put-bucket-cors --bucket "$BUCKET" --cors-configuration \
  "{\"CORSRules\":[{\"AllowedOrigins\":[\"$SITE_ORIGIN\"],\"AllowedMethods\":[\"POST\"],\"AllowedHeaders\":[\"*\"],\"MaxAgeSeconds\":300}]}"

echo "Upload API ready."
