#!/bin/bash

# Script to verify DNS configuration for api.klipz.ai

set -e

echo "🔍 Verifying DNS configuration for api.klipz.ai..."
echo ""

# Check if CNAME record exists
CNAME_RECORD=$(dig +short api.klipz.ai CNAME 2>/dev/null || echo "")

if [ -z "$CNAME_RECORD" ]; then
    echo "❌ CNAME record not found for api.klipz.ai"
    echo ""
    echo "📋 Please add this CNAME record in GoDaddy:"
    echo "   Type: CNAME"
    echo "   Name: api"
    echo "   Value: ClipIt-ClipI-KLYyN2QV2eKx-1690752772.us-east-1.elb.amazonaws.com"
    echo "   TTL: 600"
    echo ""
    echo "⏳ After adding, wait 5-30 minutes for DNS propagation."
    exit 1
fi

echo "✅ CNAME record found: $CNAME_RECORD"
echo ""

# Check if it points to the correct ALB
EXPECTED_ALB="ClipIt-ClipI-KLYyN2QV2eKx-1690752772.us-east-1.elb.amazonaws.com"

if [[ "$CNAME_RECORD" == *"$EXPECTED_ALB"* ]]; then
    echo "✅ CNAME points to correct ALB"
else
    echo "⚠️  CNAME points to: $CNAME_RECORD"
    echo "   Expected: $EXPECTED_ALB"
    echo "   Please verify the DNS record in GoDaddy."
fi

echo ""
echo "🔍 Testing HTTPS connection..."
echo ""

# Test HTTPS connection
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 https://api.klipz.ai/health 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ HTTPS is working! Status code: $HTTP_CODE"
    echo ""
    echo "🎉 Your API is accessible at: https://api.klipz.ai"
elif [ "$HTTP_CODE" = "000" ]; then
    echo "⚠️  Could not connect to https://api.klipz.ai"
    echo "   This might mean:"
    echo "   - DNS hasn't propagated yet (wait 5-30 minutes)"
    echo "   - DNS record is not configured correctly"
    echo "   - Certificate validation issue"
else
    echo "⚠️  HTTPS connection returned status code: $HTTP_CODE"
    echo "   The domain is resolving, but there might be an issue with the API."
fi

echo ""
echo "📋 To check DNS propagation status:"
echo "   dig api.klipz.ai"
echo ""
echo "📋 To test HTTPS manually:"
echo "   curl -v https://api.klipz.ai/health"

