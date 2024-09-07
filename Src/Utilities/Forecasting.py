def get_expected_demand(self):

    buyer_demand, buyer_price = 0, 0
    if isinstance(self, Firm1):
        buyer_demand, buyer_price = self.get_market_demand('capital')
        self.expected_demand =expect_demand(buyer_demand,(self.model.config.TIME_HORIZON))
        self.expected_price = expect_price(buyer_price, (self.model.config.TIME_HORIZON))
        self.expectations=[np.mean(self.expected_demand), np.mean(self.expected_price)]
        self.expectations_cache.append(self.expectations)

        if len(self.expectations_cache)>5:
            self.expectations_cache = self.expectations_cache[-5:]

        self.expectations = np.mean(self.expectations, axis=0)
    elif isinstance(self, Firm2):
        buyer_demand, buyer_price = self.get_market_demand('consumption')
        self.expected_demand = expect_demand(buyer_demand,(self.model.config.TIME_HORIZON))
        self.expected_price = expect_price(buyer_price, (self.model.config.TIME_HORIZON))
        self.expectations=[np.mean(self.expected_demand), np.mean(self.expected_price)]

    return self.expected_demand, self.expected_price

def get_market_demand(self, market_type):
    if market_type == 'labor':
        potential_buyers = [agent for agent in self.model.schedule.agents if isinstance(agent, Firm)]
        buyer_demand = [firm.labor_demand for firm in potential_buyers]
        buyer_demand = sum(buyer_demand)
        buyer_price = [firm.wage for firm in potential_buyers]
        buyer_price = np.mean(buyer_price)
    elif market_type == 'capital':
        potential_buyers = [agent for agent in self.model.schedule.agents if isinstance(agent, Firm2)]
        buyer_demand =[firm.investment_demand for firm in potential_buyers]
        buyer_demand = sum(buyer_demand)/2
        buyer_price = [firm.get_max_capital_price() for firm in potential_buyers]
        buyer_price = np.mean(buyer_price)
    elif market_type == 'consumption':
        potential_buyers = [agent for agent in self.model.schedule.agents if hasattr(agent,'consumption')]
        buyer_demand = [agent.desired_consumption for agent in potential_buyers]
        buyer_demand = sum(buyer_demand)/5
        buyer_price = [agent.expected_price for agent in potential_buyers]
        buyer_price = np.mean(buyer_price)

    else:
        raise ValueError(f"Invalid market type: {market_type}")
    avg_price = buyer_price
    self.historic_demand.append(buyer_demand)

    return buyer_demand, buyer_price