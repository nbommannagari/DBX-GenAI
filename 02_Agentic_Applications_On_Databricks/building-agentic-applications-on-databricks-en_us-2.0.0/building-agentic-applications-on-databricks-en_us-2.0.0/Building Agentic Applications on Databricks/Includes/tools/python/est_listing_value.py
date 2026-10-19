def est_listing_value(
    p1: float,
    p2: float,
    p3: str
) -> str:
    """
    Calculates value.

    Args:
        p1: price
        p2: count
        p3: market

    Returns:
        str: result
    """

    base_value = p2 * 365 * 0.7  # assume 70% occupancy
    bedroom_factor = 1 + (p1 - 1) * 0.15
    if p3.lower() == 'entire home/apt':
        multiplier = 1.2
    elif p3.lower() == 'private room':
        multiplier = 0.8
    else:
        multiplier = 0.5
    estimated = base_value * bedroom_factor * multiplier
    return f"Estimated annual revenue: ${estimated:,.2f}"
