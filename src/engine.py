import pulp
import pandas as pd

class MicrogridOptimizer:
    def __init__(self, config):
        self.config = config
        self.specs = config['current_specs']
        self.prices = config['optimization']['grid_prices_cents_kwh']
        self.inverter_kva = self.specs['inverter_kva']

    def _get_price(self, hour_of_day):
        h = hour_of_day % 24
        # Look up TOU price
        if 7 <= h < 11 or 17 <= h < 19: return self.prices['mid_peak']
        if 11 <= h < 17: return self.prices['on_peak']
        return self.prices['off_peak']

    def solve(self, load_forecast, solar_forecast, initial_soc, horizon_hours=24, force_discharge=False):
        
        T = len(load_forecast)
        time_steps = range(T)
        prob = pulp.LpProblem("Microgrid_Dispatch", pulp.LpMinimize)
        
        # --- Variables ---
        grid_import = pulp.LpVariable.dicts("Grid_Import", time_steps, lowBound=0)
        batt_charge = pulp.LpVariable.dicts("Batt_Charge", time_steps, lowBound=0, upBound=self.specs['battery_max_charge_kw'])
        batt_discharge = pulp.LpVariable.dicts("Batt_Discharge", time_steps, lowBound=0, upBound=self.specs['battery_max_discharge_kw'])
        batt_soc = pulp.LpVariable.dicts("Batt_SOC", time_steps, lowBound=0, upBound=self.specs['battery_capacity_kwh'])
        is_charging = pulp.LpVariable.dicts("Is_Charging", time_steps, cat='Binary')
        
        # *** REACTIVE POWER VARIABLE ***
        q_out = pulp.LpVariable.dicts("Reactive_kVAR", time_steps, 
                                      lowBound=-self.inverter_kva, upBound=self.inverter_kva)

        # --- Objective Function ---
        energy_cost = pulp.lpSum([grid_import[t] * self._get_price(t) for t in time_steps])
        wear_cost = pulp.lpSum([(batt_charge[t] + batt_discharge[t]) * self.specs['battery_degradation_cost_cents_kwh'] for t in time_steps])
        
        # *** REVENUE: Ancillary Service/Voltage Support (Mocked at 0.05 cents/kVARh) ***
        var_revenue = pulp.lpSum([q_out[t] * 0.05 for t in time_steps]) 
        
        prob += energy_cost + wear_cost - var_revenue

        # --- Constraints ---
        eff_c = self.specs['battery_efficiency_charge']
        eff_d = self.specs['battery_efficiency_discharge']
        BIG_M = self.specs['battery_max_charge_kw'] * 10
        
        for t in time_steps:
            # 1. Active Power Balance (P)
            prob += (grid_import[t] + solar_forecast[t] + (batt_discharge[t] * eff_d) 
                     == load_forecast[t] + batt_charge[t])
            
            # 2. Inverter Capability (P + |Q| <= S)
            prob += (batt_charge[t] + batt_discharge[t] + q_out[t]) <= self.inverter_kva
            
            # 3. SOC Physics
            prev_soc = initial_soc if t == 0 else batt_soc[t-1]
            prob += batt_soc[t] == prev_soc + (batt_charge[t] * eff_c) - batt_discharge[t]
            
            # 4. Mutual Exclusivity
            prob += batt_charge[t] <= is_charging[t] * BIG_M
            prob += batt_discharge[t] <= (1 - is_charging[t]) * BIG_M
            
            # 5. ICI Override (The Critical Constraint)
            if force_discharge and t < 6: # Force discharge for the first few hours of the event
                prob += batt_discharge[t] >= self.specs['battery_max_discharge_kw'] * 0.95
                prob += grid_import[t] <= 10.0 # Force import near zero

        # 6. Cycle Constraint (Return to starting SOC for the next day/cycle)
        prob += batt_soc[T-1] == initial_soc
        
        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Extract Results
        results = []
        for t in time_steps:
            results.append({
                "Hour": t,
                "Load_kW": load_forecast[t],
                "Solar_kW": solar_forecast[t],
                "Grid_Import_kW": pulp.value(grid_import[t]),
                "Batt_SOC_kWh": pulp.value(batt_soc[t]),
                "Batt_Charge_kW": pulp.value(batt_charge[t]),
                "Batt_Discharge_kW": pulp.value(batt_discharge[t]),
                "Reactive_kVAR": pulp.value(q_out[t]),
                "Price_cents": self._get_price(t)
            })
            
        return pd.DataFrame(results), pulp.value(prob.objective), pulp.value(batt_soc[0])