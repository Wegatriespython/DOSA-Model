class Config:
    def __init__(self):
        # Common parameters
        self.INITIAL_CAPITAL = 100
        self.INITIAL_PRODUCTIVITY = 1
        self.INITIAL_PRICE = 1
        self.INITIAL_WAGE = 1
        self.INITIAL_INVENTORY = 2
        self.INITIAL_DEMAND = 10
        self.INITIAL_SKILLS = .1
        self.MARKUP_RATE = 0.2
        self.INVENTORY_THRESHOLD = 15
        self.PRODUCTION_FACTOR = 0.1
        self.WAGE_OFFER_FACTOR = 1.1
        self.SKILL_GROWTH_RATE = 0.01
        self.SKILL_DECAY_RATE = 0.005
        self.JOB_LOSS_PROBABILITY = 0.05
        self.MINIMUM_WAGE = 0.5
        self.CONSUMPTION_PROPENSITY =1
        self.INITIAL_CONSUMPTION = 0
        # Firm1 specific parameters
        self.FIRM1_INITIAL_CAPITAL = 100
        self.FIRM1_INITIAL_RD_INVESTMENT = 0
        self.FIRM1_RD_INVESTMENT_RATE = 0.1
        self.FIRM1_INVENTORY_THRESHOLD = 10
        self.INNOVATION_PROBABILITY = 0.1
        self.PRODUCTIVITY_INCREASE = 0.1
        
        # Firm2 specific parameters
        self.FIRM2_INITIAL_CAPITAL = 1000
        self.FIRM2_INITIAL_INVESTMENT = 0
        self.FIRM2_INITIAL_INVESTMENT_DEMAND = 10
        self.FIRM2_INITIAL_DESIRED_CAPITAL = 0
        self.FIRM2_MACHINE_OUTPUT_PER_PERIOD = 10
        self.FIRM2_INVENTORY_THRESHOLD = 20

        # Simulation parameters
        self.INITIAL_WORKERS = 10
        self.INITIAL_CAPITAL_FIRMS = 2  # This was previously INITIAL_FIRM1S
        self.INITIAL_CONSUMPTION_FIRMS = 5  # This was previously INITIAL_FIRM2S
        self.SIMULATION_STEPS = 10

# Create a global configuration object
config = Config()