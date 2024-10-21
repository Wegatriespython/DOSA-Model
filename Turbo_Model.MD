### Turbo Model 

This model is intended to couple the optimisation functions for firms and workers with the auction system. 
Compared to the full model there is no purely simulation behaviors. For example workers and firms make and sell things but they don't update their accounts. Firms dont have an employee list and workers dont have a function like consume. 

Instead the goal of this model is the following : 

Based on the auction system firms and workers update their supply and demand functions. 
Then they run through the best response function for prices based on the previous round's bid, ask, bid_max, ask_min, supply, demand. 

Our goal is to see if we can acheive a fixed point equilibrium in this market economy. 