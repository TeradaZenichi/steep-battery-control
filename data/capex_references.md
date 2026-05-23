# BESS and EV Battery CAPEX Reference

## Summary of Chosen Values

| Asset | capex (EUR, total) | Emax (kWh) | EUR/kWh | ncycles | Degradation (EUR/kWh throughput) |
|-------|-------------------|------------|---------|---------|----------------------------------|
| BESS  | 750               | 5          | 150     | 6000    | 0.025                            |
| EV    | 6000              | 40         | 150     | 671     | 0.224                            |

BESS degradation is **8.9× cheaper** per kWh throughput than EV — consistent with
stationary LFP batteries being designed for daily cycling (6000 cycles) vs. EV
batteries experiencing accelerated degradation under V2G (671 additional cycles).

## BESS: 150 EUR/kWh (total 750 EUR for 5 kWh)

### Battery pack cost (cells + BMS, excluding inverter/installation)

- **BNEF 2025**: Global average LFP battery pack price fell to **$81/kWh**; stationary
  storage packs specifically reached **$70/kWh** — the lowest of any segment.
  European prices are ~56% higher than global average due to import dependence,
  putting European LFP stationary packs at **~$109/kWh (~€100/kWh)**.
  Source: [BloombergNEF, Dec 2025](https://about.bnef.com/insights/clean-transport/lithium-ion-battery-pack-prices-fall-to-108-per-kilowatt-hour-despite-rising-metal-prices-bloombergnef/)

- **BNEF 2024**: LFP pack prices dropped 20% YoY to **$115/kWh** globally.
  Source: [BloombergNEF, Nov 2024](https://about.bnef.com/insights/commodities/lithium-ion-battery-pack-prices-see-largest-drop-since-2017-falling-to-115-per-kilowatt-hour-bloombergnef/)

- **NREL ATB 2024**: Residential BESS reference system is 5 kW / 12.5 kWh.
  Battery pack cost component: **$283/kWh** (2022 USD, includes markup).
  Total installed system cost is much higher ($700–1300/kWh) but includes
  inverter, BOS, installation labor, permitting, and profit margins — not
  relevant for degradation cost modeling where only cell replacement matters.
  Source: [NREL ATB 2024 – Residential Battery Storage](https://atb.nrel.gov/electricity/2024/residential_battery_storage)

### Justification for 150 EUR/kWh

The degradation cost model represents the cost of **replacing battery cells at
end-of-life**, not the full system installation. The relevant metric is the
battery pack cost (cells + BMS). Using 2024-2025 European pricing:

- Global LFP pack: $70–115/kWh (2024–2025)
- European premium: +56% → $109–180/kWh → **€100–165/kWh**
- Chosen value: **150 EUR/kWh** (conservative mid-range for European market)
- Total for 5 kWh system: **750 EUR**

## EV: 150 EUR/kWh (total 6000 EUR for 40 kWh)

### EV battery replacement cost

- **Industry average 2024-2025**: EV battery replacement costs range from
  **$120–250/kWh** depending on vehicle and chemistry, with total replacement
  costs of $5,000–$20,000.
  Source: [Recurrent Auto, 2025](https://www.recurrentauto.com/research/costs-ev-battery-replacement)

- **BNEF 2025**: EV battery pack prices averaged **$108/kWh** globally
  (all chemistries), with LFP packs at $81/kWh.
  Source: [BloombergNEF, Dec 2025](https://about.bnef.com/insights/clean-transport/lithium-ion-battery-pack-prices-fall-to-108-per-kilowatt-hour-despite-rising-metal-prices-bloombergnef/)

### V2G degradation economics

- **Salinas et al. (2025)**: V2G increases battery degradation by 9–14% over
  10 years. Cyclic degradation contributes 25% of total degradation with V2G
  (vs. 15% without). Economic compensation required: **€132/MWh** (2030 scenario)
  = €0.132/kWh of V2G energy flow.
  Source: [Applied Energy, Vol. 377, 2025](https://www.sciencedirect.com/science/article/pii/S0306261924019299)

### Justification for 150 EUR/kWh

For V2G degradation modeling, the relevant cost is the marginal cost of
battery cell replacement caused by V2G cycling. Using European replacement
pricing:

- Pack replacement: $120–250/kWh → €110–230/kWh
- Chosen value: **150 EUR/kWh** (mid-range, same as BESS for simplicity)
- Total for 40 kWh pack: **6000 EUR**
- With ncycles=671 (V2G additional cycles): 6000/(40×671) = **0.224 EUR/kWh**

This is consistent with the V2G compensation estimate of €0.132/kWh from
Salinas et al. — the degradation cost is higher because it includes the full
replacement cost, while compensation estimates factor in remaining battery
value and other revenue streams.

## Impact on RL Training

With these corrected values:
- BESS arbitrage: spread 0.057 EUR/kWh − degradation 0.025 = **+0.032 EUR/kWh net profit**
- The agent has a clear economic incentive to use BESS for tariff arbitrage
- EV V2G remains expensive (0.224 EUR/kWh) — agent correctly avoids V2G for
  pure arbitrage but uses it to avoid fast_tariff (3× grid tariff) penalties
