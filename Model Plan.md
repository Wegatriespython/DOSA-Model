# Agent-Based Economic Model

## Model Overview
This model simulates an economy with three interconnected sectors: capital goods production, consumption goods production, and a labor market. The model uses an agent-based approach with discrete time steps and a fixed number of agents in each sector.

## Sectors and Markets

1. **Capital Goods Sector (Sector 1)**
   - Produces capital goods using labor as input
   - Sellers in the capital market
   - Buyers in the labor market

2. **Consumption Goods Sector (Sector 2)**
   - Produces consumption goods using capital and labor as inputs
   - Buyers in the capital market
   - Buyers in the labor market
   - Sellers in the consumption market

3. **Labor Market**
   - Workers are sellers
   - Both Sector 1 and Sector 2 firms are buyers

## Production Functions
- Both firm types use Cobb-Douglas production functions
- Capital goods firms: capital elasticity of 0
- Consumption goods firms: capital elasticity of 0.5

## Agent Decision-Making

### Firms
- Solve intertemporal profit maximization problems for production and investment decisions
- Use simple AR projections for future prices and quantities
- Intertemporal decision-making

### Workers
- Solve utility maximization problems for labor supply, consumption, and savings
- Cobb-Douglas utility function between consumption and leisure
### Expectations 
- Used AR expectations initially, but now using adaptive expectations for firms. 
- Optimisation done based on expectations

## Market Clearing Mechanism

### Inputs
- Buyers: quantity demanded, desired price, max price
- Sellers: quantity supplied, desired price, min price

### Process
1. Round 1: Transactions occur at desired prices
2. Round 2: Adjustments based on market conditions
   - Excess demand: Sellers have advantage, buyers use max price
   - Excess supply: Buyers have advantage, sellers use min price

### Price Adjustment Algorithms

#### Buyers
- If demand not met: Increase desired price
- If demand met:
  - Clearing price > desired price: Increase desired price
  - Clearing price < desired price: Decrease desired price

#### Sellers
- If supply not cleared: Lower desired price
- If supply cleared:
  - Clearing price > desired price: Increase desired price
  - Clearing price < desired price: Decrease desired price

### Heuristic Adjustment
- Buyers: Based on actual_consumption / desired_consumption
- Sellers: Based on actual_sales / desired_sales
- Both: Consider clearing_price / desired_price ratio

## Model Equilibrium
The model aims to produce endogenous competitive prices in each market. Agents have limited information:
- Clearing prices in each market
- Aggregate expected demand and supply
- Latent market demand, supply, and price

## Open Questions
- Is this mechanism sufficient to produce an equilibrium with competitive markets?
- How does the limited information available to agents affect market outcomes?
