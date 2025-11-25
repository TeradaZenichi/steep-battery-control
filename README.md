# steep-battery-control

Smart home battery control system with energy management.

## Units Convention

All costs and penalties in this project are based on **energy (kWh)**, not power (kW):

- **Tariffs** (`tar_s`, `tar_w`, `tar_sw`, `tar_flat`, `tar_tou`): €/kWh
- **Time-of-use rates** (`tou`): €/kWh  
- **Flat rate** (`flat_rate`): €/kWh
- **Net penalty** (`net_penalty`): €/kWh

### Power vs Energy Parameters

- **Power parameters** (in kW): `Pmax`, `Pmax_import`, `Pmax_export`, `Pmax_c`, `Pmax_d`
- **Energy capacity parameters** (in kWh): `Emax`

### Computing Total Cost

To compute the total cost for a given time period, multiply the tariff (€/kWh) by the energy consumed (kWh):

```
Total Cost = Tariff [€/kWh] × Energy [kWh]
```

For time-series data with a fixed timestep, convert power to energy:

```
Energy [kWh] = Power [kW] × Timestep [h]
```