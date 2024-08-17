    def predict_demand(self):
        if not self.is_trained:
            return np.mean(self.historic_sales)  # Fallback to historical average if not trained

        features = self.prepare_features()
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.demand_predictor.predict(features_scaled)[0]

        return max(prediction, 0)  # Ensure non-negative demand

    def prepare_features(self):
        return np.array([
            self.capital,
            len(self.workers),
            self.productivity,
            self.price,
            self.inventory,
            self.budget,
            np.mean(self.historic_sales),
            np.std(self.historic_sales),
            self.model.get_average_wage(),  # Updated
            self.model.get_average_capital_price(),  # Updated
            self.model.get_average_consumption_good_price(),  # Updated
            self.get_market_demand(self.get_market_type())
        ])

    def train_demand_predictor(self):
        if len(self.historic_sales) < 10:  # Need some history to train
            return

        X = np.array([self.prepare_features() for _ in range(len(self.historic_sales) - 5)])
        y = np.array(self.historic_sales[5:])  # Predict next period's sales

        X_scaled = self.scaler.fit_transform(X)
        self.demand_predictor.fit(X_scaled, y)
        self.is_trained = True

        logging.info(f"Firm {self.unique_id} - Demand predictor trained. Coefficients: {self.demand_predictor.coef_}")
        def train_and_save_models(self):
            for firm_id, data in self.data_collection.items():
                X = np.array([d[0] for d in data])
                y_demand = np.array([d[1] for d in data])
                y_sales = np.array([d[2] for d in data])

                model_demand = LinearRegression()
                model_demand.fit(X, y_demand)

                model_sales = LinearRegression()
                model_sales.fit(X, y_sales)

                print(f"Firm {firm_id} - Trained demand model coefficients: {model_demand.coef_}")
                print(f"Firm {firm_id} - Trained sales model coefficients: {model_sales.coef_}")

                joblib.dump(model_demand, f'demand_predictor_firm_{firm_id}.joblib')
                joblib.dump(model_sales, f'sales_predictor_firm_{firm_id}.joblib')
        def analyze_predictions(self):
            for agent in self.schedule.agents:
                if isinstance(agent, (Firm1, Firm2)):
                    agent.analyze_prediction_accuracy()
