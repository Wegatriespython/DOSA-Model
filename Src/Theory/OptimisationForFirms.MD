# Mathematical Formulation of Profit Maximization Problem

## Objective Function

Maximize the discounted sum of profits over T periods:

$$
\max_{L_t, K_t, Q_t, I_t, S_t} \sum_{t=0}^{T-1} \frac{1}{(1+r)^t} (P_t S_t - w L_t - p_K (K_t - K_{t-1}) - \delta P_t I_t)
$$

Where:
- $L_t$: Labor input at time t
- $K_t$: Capital input at time t
- $Q_t$: Production at time t
- $I_t$: Inventory at time t
- $S_t$: Sales at time t
- $P_t$: Expected price at time t
- $w$: Wage rate
- $p_K$: Price of capital
- $r$: Discount rate
- $\delta$: Depreciation rate

## Constraints

1. Production function (Cobb-Douglas):
   $$Q_t = A K_t^\alpha L_t^{1-\alpha}$$
   Where $A$ is the productivity factor and $\alpha$ is the capital elasticity.

2. Inventory balance:
   $$I_t = I_{t-1} + Q_t - S_t$$

3. Sales constraint (demand):
   $$S_t \leq D_t$$
   Where $D_t$ is the expected demand at time t.

4. Sales constraint (inventory):
   $$S_t \leq I_t + Q_t$$

5. Budget constraint:
   $$w L_t + p_K (K_t - K_{t-1}) \leq B$$
   Where $B$ is the available budget.

6. Non-negativity constraints:
   $$L_t, K_t, Q_t, I_t, S_t \geq 0$$

7. Initial conditions:
   $$K_{-1} = K_0, I_{-1} = I_0$$
   Where $K_0$ and $I_0$ are given initial values for capital and inventory.

## Additional Considerations

- The model assumes perfect foresight for prices and demand over the planning horizon.
- The optimization problem is solved for each firm independently, assuming no strategic interactions between firms.
- The solution provides optimal paths for labor, capital, production, inventory, and sales over the planning horizon.
