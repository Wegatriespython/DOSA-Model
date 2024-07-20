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
        self.skills = config.INITIAL_SKILLS
        self.offers = []  # List to store wage offers
        self.consumption = config.INITIAL_CONSUMPTION  # Initialize the consumption attribute

    def update_state(self):
        if self.employed:
            self.skills *= (1 + config.SKILL_GROWTH_RATE)
            if random.random() < config.JOB_LOSS_PROBABILITY:
                self.employed = False
                self.employer = None
                print(f"Worker {id(self)} lost their job")
        else:
            self.skills *= (1 - config.SKILL_DECAY_RATE)
        self.wage = max(self.wage * (1 + random.uniform(-0.05, 0.05)), config.MINIMUM_WAGE)


    def apply_for_jobs(self, firms):
        """
        Simulates the worker applying for jobs.
        This is a simplified version for now.
        """
        for firm in firms:
            wage = firm.get_wage_offer(self)  # Get a wage offer from the firm
            self.offers.append(WageOffer(self, firm, wage))  # Store the offer

    def get_wage(self):
        """
        Returns the worker's current wage.
        """
        return self.wage

    def set_wage(self, new_wage):
        """
        Sets the worker's wage.
        """
        self.wage = new_wage

    def get_skills(self):
        """
        Returns the worker's current skills.
        """
        return self.skills

    def get_employed(self):
        """
        Returns the worker's current employment status.
        """
        return self.employed

    def set_employed(self, is_employed):
        """
        Sets the worker's employment status.
        """
        self.employed = is_employed

    def get_employer(self):
        """
        Returns the worker's current employer.
        """
        return self.employer

    def set_employer(self, employer):
        """
        Sets the worker's employer.
        """
        self.employer = employer
    def consume(self, firms):
        print(f"DEBUG: Worker {id(self)} starting consumption")
        print(f"DEBUG: Worker employed: {self.employed}, wage: {self.wage}")
        
        if self.employed and self.wage > 0:
            consumption = self.wage * config.CONSUMPTION_PROPENSITY
            print(f"DEBUG: Calculated consumption: {consumption}")
            
            selected_firm = random.choice(firms)
            print(f"DEBUG: Selected firm {id(selected_firm)} for consumption")
            
            if selected_firm.inventory >= consumption:
                selected_firm.inventory -= consumption
                self.wage -= consumption
                self.consumption = consumption
                print(f"DEBUG: Consumption successful. New wage: {self.wage}, Firm inventory: {selected_firm.inventory}")
            else:
                print(f"DEBUG: Consumption failed. Firm inventory ({selected_firm.inventory}) < consumption ({consumption})")
        else:
            print(f"DEBUG: Worker unable to consume. Employed: {self.employed}, Wage: {self.wage}")
        
        print(f"DEBUG: Worker {id(self)} finished consumption attempt")