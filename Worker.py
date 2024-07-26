# Worker.py
from Config import config
import random
import math
from Wageoffer import WageOffer

class Worker:
    def __init__(self):
        self.employed = False
        self.employer = None
        self.wage = config.INITIAL_WAGE
        self.savings = config.INITIAL_SAVINGS
        self.skills = config.INITIAL_SKILLS
        self.offers = []
        self.consumption = config.INITIAL_CONSUMPTION
        self.satiated = False

    def update_state(self):
        changes = []
        changes.extend(self.update_skills())
        if self.employed:
            changes.append((self, 'wage', self.employer.get_wage_offer(self)))
        else:
            changes.append((self, 'wage', max(1, self.wage * (1 - config.SKILL_DECAY_RATE))))
        return changes

    def update_skills(self):
        if self.employed:
            new_skills = self.skills * (1 + config.SKILL_GROWTH_RATE)
        else:
            new_skills = self.skills * (1 - config.SKILL_DECAY_RATE)
        return [(self, 'skills', new_skills)]

    def apply_for_jobs(self, firms):
        self.offers = []
        for firm in firms:
            wage = firm.get_wage_offer(self)
            self.offers.append(WageOffer(self, firm, wage))

    def consume(self, firms):
        desired_consumption = min(self.wage * config.CONSUMPTION_PROPENSITY, self.savings)
        return desired_consumption

    def update_savings_and_consumption(self, actual_consumption, price):
        self.savings -= actual_consumption * price
        self.consumption += actual_consumption