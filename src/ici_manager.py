import numpy as np

class ICIManager:
    def __init__(self, historical_peaks_mw, threshold_factor=0.95):
        """
        Manages Global Adjustment (GA) mitigation logic for Ontario Class A.
        Checks if the system-wide demand forecast exceeds a critical threshold 
        to trigger battery discharge.
        """
        self.critical_peak = min(historical_peaks_mw)
        self.trigger_threshold = self.critical_peak * threshold_factor
        
    def check_trigger(self, current_ontario_demand_forecast):
        """
        Returns True if we should ignore price and discharge now to avoid GA costs.
        """
        # Note: This is a simplification. Real systems use proprietary forecast algorithms.
        return current_ontario_demand_forecast > self.trigger_threshold

    def calculate_potential_savings(self, peak_reduction_kw):
        """
        Estimates annual savings based on avoided GA costs (approx $500k/MW).
        """
        SAVINGS_PER_MW = 500000.0
        return (peak_reduction_kw / 1000.0) * SAVINGS_PER_MW