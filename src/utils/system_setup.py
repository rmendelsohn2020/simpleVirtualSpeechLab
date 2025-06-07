def params_to_eqnvars(self, channels=["aud_", "som_"]):
    #Setup for noise injections/knockouts
    for channel in channels:
        L = getattr(self, f"{channel}L")
        L_del = getattr(self, f"{channel}L_del")
        Kf = getattr(self, f"{channel}Kf")
        K = getattr(self, f"{channel}K")
        K_del = getattr(self, f"{channel}K_del")
        
        #Create new variables
        setattr(self, f"{channel}L1", L)
        setattr(self, f"{channel}L_del", L_del)
        setattr(self, f"{channel}Kf1", Kf)
        setattr(self, f"{channel}Kf2", Kf)
        setattr(self, f"{channel}K1", K)
        setattr(self, f"{channel}K2", K)
        setattr(self, f"{channel}K_del", K_del)