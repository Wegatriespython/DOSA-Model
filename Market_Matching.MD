
# Market Matching 

This is a two round auction game. There are multiple buyers and sellers. Buyers sumbit their demand, bid_price, max_bid_price. Sellers submit their supply, ask_price, min_ask_price. 

In the first round a trade will happen when a buyer's bid >= a seller's ask. Each transaction is atomic. There is no central clearing price. 

When no trades are possible, round two happens. 

Here the aggregate demand and supply are calculated. If Demand > Supply then its seller's advantage. 
If Supply > Demand it's the buyer's advantage. 

In case it is seller's advantage, a trade will happen when a buyer's max_bid_price >= seller's ask_price. Otherwise trades will not occur. 

In the case it is a buyer's advantage a trade will happen when a buyer's bid_price >= seller's min_ask_price. 


The prices at which a transaction clears is at the mid-point of the respective bid and ask price. 

Implementation: 
 Trades are evaluated through sorted arrays. The buyer highest bid trades against the seller with the lowest ask price. 


Strategy : 

Now there are several scenarious which affect the case of the game. Buyer's want to meet their full demand and seller's want to clear their full supply. 

At higher prices sellers will increase supply and buyers will decrease their demand. 

At lower prices seller's will decrease their supply and buyers will increase their demand. 

Outside of the case where demand=supply, there is free money on the table for one party. Under excess demand seller's can gain by chargin up and under excess supply buyer's can gain by paying less. 

Buyer's and Seller's can observe the full data for previous rounds and the game is played for n-rounds. The decisions for all buyers and seller's are made simultaneously. 

Risk : demand and supply are sums, all the prices(bid,ask, max_bid, min_ask) players observe are averages from the previous round. THere is no guarantee the next round prices will be the same.

Players have their private reservation prices which are not neccessarily == max_bid or min_ask. 

For buyer's if max_bid > pvt_res_price they are at a disadvantage and hence must offer their pvt_res_price to maximize their chances. 
if max_bid < pvt_res_price they might be safe, but if the next round sees a higher max_bid they would lose out on the trade. 

For seller's if min_ask < pvt_res_price they are at a disadvantage because they will be undercut by competition. 

if min_ask > pvt_res_price they might be safe, but if the next round sees a lower min_ask they would lose out on the trade.  

# Worker/Consumer's Problem :

1. Given set of prices for consumption and wages determine optimal consumption, labor supply and savings. 
2. Using knowledge from market matching find lowest bid price to fulfill demand and maximize utility. 

# Firm's Problem : 

1. Given a set of prices for consumption and wages, determine optimal production, and labor demand.
2. Using knoweledge from market matching find highest ask price to clear supply and maximize profit. 

Optimisation takes prices as a given for both. 

For bid's to work, true reservation prices need to be found. 

We don't solve for that, instead we give a heuristic upper and lower bound for firms and buyers. 

The idea is iteratively price-discovery through the market matching auction will lead to the discovery of the true reservation prices. 


# Early Solution Attempts 

## Nash Stability : 

Buyer's Dilemma : 
    Bid Honest : State max_bid = pvt_reservation_price 
    Bid Deception : State max_bid < pvt_reservation_price 
Seller's Dilemma : 
    Bid honest : State min_bid = pvt_reservation_price
    Bid Deception : State max_bid > pvt_reservation_price.

### One Shot Symmetric Case

If player's max price is the marginal utlity of consumption then payoff for H and recieving full consumption is H. 
Same price by both means both share half.

Let -e be marginal disutility of consumption. 

Let u be the utility of successful Deception under consumption. u = H - D

In the symmetric case where both have the same pvt_reservation_prices. H > D 

Two buyer's, Demand > Supply (Zero-Sum)
1.
B1 : Honest, B2 : Deception
Payoff (0,-e)
2.
B1 : Honest, B2 : Honest
Payoff (-e/2, -e/2)
3. 
B1 : Deception B2: Honest 
Payoff (-e, 0)
4.
B1 : Deception, B2 : Deception 
Payoff (-e/2 + u, -e/2 + u) 

When u > e/2 : Nash Equilibria might seem Deception, Deception but, there is no guarantee both will choose the same D. The buyer with higher D will get all the utility. Iteratively there is a race to the top as buyers will iteratively converge to H, H. 

Nash Equilibira under all cases H, H. 

### One Shot Assymmetric Case with Certainity 

Everything same as the symmetric case except, One of the buyers has a differnt pvt_reservation_price 

H1 != H2
Buyer with higher max_bid knows it. 

Assume H1 > H2 (no loss of generality)
Dominant Strategy
    B1 Deceive at bid price = H2 + infinitesimal. 
    B2 always loses 

### One Shot Assymmetric Case with Uncertainity

H1 != H2 
Lets say the gap is known to be episilon. 

p(higher bidder) = 0.5, coin toss between either bidder on having the higher bid. 

Case 1:
If B1 is higher then still they don't known the exact bid of B2. Iteratively setting their bid just below their max_bid might be the safest choice D1 = H1 - epsilon. B2 will choose H2
Case 2: 
If B2 is higher they choose D2 = H2 - epsilon. B1 will choose 

Mixed strategy eqilibria = Honest price * 0.5  + Honest price - epsilon *0.5 


Nash Equilibrium:
1. If u > e/2: (D, D) is the unique Nash Equilibrium
Both buyers will choose to be deceptive as it provides a higher expected payoff regardless of the other's strategy.
If 0 < u < e/2: Mixed Strategy Equilibrium
There's no pure strategy Nash Equilibrium. Buyers will randomize between H and D.
3. If u ≤ 0: (H, H) is the unique Nash Equilibrium
Being honest provides a higher expected payoff regardless of the other's strategy.

### Many Shot Symmetric Case 