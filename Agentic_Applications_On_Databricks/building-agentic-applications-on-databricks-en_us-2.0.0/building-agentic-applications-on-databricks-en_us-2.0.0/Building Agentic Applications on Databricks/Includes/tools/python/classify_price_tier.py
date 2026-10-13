def classify_price_tier(price: float) -> str:
    """Classify an Airbnb listing price into a market tier based on SF pricing distribution.

    Uses hardcoded SF Airbnb market thresholds to label a nightly price as
    Budget, Mid-Range, Premium, or Luxury, and returns the corresponding
    percentile band for context.

    Useful for questions like: 'Is $180/night a good deal in SF?',
    'What tier does this listing fall into?'

    Returns a JSON object with tier, percentile_band, and a short interpretation.

    Args:
        price: Nightly listing price in USD.
    """
    import json

    tiers = [
        (100,  "Budget",    "bottom 25%",  "Well below the SF market average."),
        (175,  "Mid-Range", "25–50%",      "Around the SF market median."),
        (300,  "Premium",   "50–75%",      "Above average for SF listings."),
        (float("inf"), "Luxury", "top 25%", "Among the most expensive listings in SF."),
    ]

    if price < 0:
        return json.dumps({"error": "Price must be a positive value."})

    for threshold, tier, band, interpretation in tiers:
        if price < threshold:
            return json.dumps({
                "price": price,
                "tier": tier,
                "percentile_band": band,
                "interpretation": interpretation
            }, indent=2)