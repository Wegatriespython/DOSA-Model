
### Extension
Okay perfect job. I am actually looking to make the following extensions while preserving compatibility with the model.

They are :

Differential capital and labor producitivity : The firm gets inouts on the average capital and labor productivity in the economy and its own capital and labor.

Carbon Intensity of production: This is a parameter that is given to each piece of capital stock. Assume this used as part of a production function to calculate the emissions of the firm. Again the firm is given the average carbon intensity of new capital in the economy as well as its own carbon_intensity of its capital stock.

Carbon tax: This is a tax that is levied on the emissions of the firm.

There are expectations for future values of labor and capital productivity, carbon intensity of capital and the carbon tax rate.Which are given for the full planning horizon.

Now the firm needs to take this carbon tax into account in formulating its investment and production decisions. It can add another variable to mark the productivity, carbon_intensity thresholds for it to buy and sell. It can lay off workers with lower productivity and higher carbon intensity. It can also invest in new capital with lower carbon intensity.

Alternatively it can retain high-carbon capital if it is profitable to do so.

Now I understand that in total this is a fair bit of complexity to be added and the problem will have to be forumulated very cleverly to make it feasible. You do not have to code it at this stage. Rather I would like you to take our existing profit maximization problem as a starting point and mathematically formulate the problem with the above extensions. I will then take it from there.

You can choose to prioritize which extensions you want to extend based on feasibility, complexity and high level relavence to incorporating carbon tax into firm decision making. I am happy to compromise on the extension plan as long as a good viable path to feasible implementation is laid out.

## Model Description
I have an agent based model w 3 sectors. Well 2 and a labor market. Sector is firms which make capital goods from labor as input. Sector 2 is consumption good firms which produce consumption good with capital and labor as input. Sector 1 firms are sellers and Sector 2 firms are buyers in capital market. Both firm types are buyers in labor market. And Sector 2 firms are sellers and workers are buyers in consumption market.

Now both firms use a cobb-douglas production function. Captial goods firms have a capital elasticisty of 0. And consumption good firms have a capital elasticity of 0.5.

Firms solve a profit maximization problem to determine their production and investment decisions. This is intertemporal and they solve using simple AR projections for fututre prices and quantities.

Workers solve a utility maximization problem to determine their labor supply, consumption and savings.

Workers have cobb-douglas utility function between consumption and leisure.

The model iterates over discrete timesteps with a fixed number of agents in each sector. Now to achieve equilibrium in the model, I have a market clearing condition for each market. Buyers provide the quantity demanded, desired_price, max_price and sellers provide the quantity supplied desired price, min_price. The fucntion works by matching buyers and sellers in each market in a decentralised way. When buyer_price > seller_price, the transaction is made at the clearing price. In round 1 transactions occur using the desired prices of both parties. In round 2, the advantage is checked, if there is excess demand, sellers have an advantage, buyers swap their desired price for the max price while sellers keep their desired price. If there is excess supply, buyers keep their desired price and sellers swap their desired price for the min price.

The idea then is to produce endogenous competitive prices in each market. In general buyers follow this algoritm for pricing:
  If demand not met by supply increase desired_price.
  If demand met by supply:
    If clearing price > desired price, increase desired price. (When transactions are occuring in Round 2 at disadvantage, so that buyers can get a better deal by increasing their desired price, which either makes transactions occur in round 1 at a lower price or lowers demand reducing advantage of sellers in round 2)
    If clearing price < desired price, decrease desired price. (When transactions are occuring in Round 1, so that buyers can get a better deal by decreasing their desired price, which either makes transactions occur in round 1 at a lower price)

Sellers follow a symmetric algorithm.
If supply does not clear lower desired price.
If supply clears:
  If clearing price > desired price, increase desired price. (Profits on the table)
  If clearing price < desired price, decrease desired price. (Disadvantage Round 2 transactions, either catch them in round 1 or lower supply and stop disadvantage in round 2)

Now there is a heuristic adjustment size determination. The ratio of actual_consumption/desired_consumption is used for buyers, and actual_sales/desired_sales is used by sellers. Additonally the ratio of clearing_price/desired_price is used by both buyers and sellers.

Now I wonder if this is sufficient to produce an equilibirium with competitive markets. Firms and workers are not given any explicit information on the prescence of other firms or workers. They only know the clearing prices in each market, aggregate expected demand, aggregate expected supply. They also know the latent market demand and supply(what could theoretical clear) and price.
