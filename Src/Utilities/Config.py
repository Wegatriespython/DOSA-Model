#test Disables

class Config:
    def __init__(self):
        # Common parameters
        #

        self.MAX_WORKING_HOURS = 16
        self.TIME_HORIZON = 100
        self.INITIAL_SALES = 1
        self.INITIAL_PRODUCTIVITY = 1
        self.INITIAL_PRICE = 1
        self.INITIAL_WAGE = 0.0625 # 1/16
        self.DEPRECIATION_RATE = 0.00
        self.INITIAL_SKILLS = .1
        self.DISCOUNT_RATE = 0.05
        self.INITIAL_SAVINGS = 15
        self.MIN_CONSUMPTION = 1
        self.PRODUCTION_FACTOR = 0.1
        self.WAGE_OFFER_FACTOR = 1.1
        self.MINIMUM_WAGE = 0.0625
        self.CONSUMPTION_PROPENSITY =1
        self.INITIAL_CONSUMPTION = 1
        self.TOTAL_FACTOR_PRODUCTIVITY = 1.0
        self.CAPITAL_ELASTICITY_FIRM2 = 0.5
        self.CAPITAL_ELASTICITY_FIRM1 = 0.1
        self.SKILL_GROWTH_RATE = 0.01
        self.SKILL_DECAY_RATE = 0.005
        # Firm1 specific parameters
        self.FIRM1_INITIAL_CAPITAL = 20
        self.FIRM1_INITIAL_INVENTORY = 5
        self.FIRM1_INITIAL_DEMAND = 3
        self.FIRM1_INITIAL_RD_INVESTMENT = 0
        self.FIRM1_RD_INVESTMENT_RATE = 0
        self.INNOVATION_ATTEMPT_PROBABILITY = 0.1  # Probability of a successful innovation attempt
        self.PRODUCTIVITY_INCREASE_PROBABILITY = 0.5  # Probability that a successful innovation increases productivity
        self.PRODUCTIVITY_INCREASE = 0.005  # Reduced from 0.1 to 0.05 for more gradual growth
        self.INITIAL_RELATIVE_PRICE_CAPITAL = 3.0  # Initial price of capital goods relative to consumption goods
        self.INITIAL_RELATIVE_PRICE_LABOR = 1.0  # Initial price of labor relative to consumption goods
        # Firm2 specific parameters
        self.FIRM2_INITIAL_CAPITAL = 6
        self.FIRM2_INITIAL_INVENTORY = 2
        self.FIRM2_INITIAL_DEMAND = 6
        self.FIRM2_INITIAL_INVESTMENT_DEMAND = 1
        self.FIRM2_INITIAL_DESIRED_CAPITAL = 5
        self.FIRM2_MACHINE_OUTPUT_PER_PERIOD = 10
        self.INVENTORY_THRESHOLD = 20

        # Simulation parameters
        self.INITIAL_WORKERS = 10
        self.INITIAL_CAPITAL_FIRMS = 2  # This was previously INITIAL_FIRM1S
        self.INITIAL_CONSUMPTION_FIRMS = 3  # This was previously INITIAL_FIRM2S
        self.SIMULATION_STEPS = 100

# Create a global configuration object
config = Config()
