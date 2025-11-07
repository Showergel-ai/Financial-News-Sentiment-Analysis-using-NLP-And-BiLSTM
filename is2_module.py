"""
IS-II Module - Portfolio Optimization using Particle Swarm Optimization (PSO)
"""

import numpy as np
import pandas as pd
import random
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
from utils.data_fetcher import DataFetcher
from utils.logger import setup_logger, log_data_processing, log_model_action

class Particle:
    """Particle class for PSO algorithm."""
    
    def __init__(self, dimension: int, bounds: List[Tuple[float, float]], min_weight: float = 0.01):
        """
        Initialize a particle.
        
        Args:
            dimension (int): Number of dimensions (assets)
            bounds (List[Tuple[float, float]]): Bounds for each dimension
            min_weight (float): Minimum weight for each asset (default 1%)
        """
        self.dimension = dimension
        self.bounds = bounds
        self.min_weight = min_weight
        
        # Initialize position (portfolio weights) with minimum constraint
        self.position = np.random.uniform(min_weight, 1, dimension)
        self.position = self.position / np.sum(self.position)  # Normalize to sum to 1
        
        # Initialize velocity
        self.velocity = np.random.uniform(-0.1, 0.1, dimension)
        
        # Initialize best position and fitness
        self.best_position = self.position.copy()
        self.best_fitness = float('-inf')
        self.fitness = float('-inf')
    
    def update_velocity(self, global_best_position: np.ndarray, w: float, c1: float, c2: float):
        """
        Update particle velocity.
        
        Args:
            global_best_position (np.ndarray): Global best position
            w (float): Inertia weight
            c1 (float): Cognitive parameter
            c2 (float): Social parameter
        """
        r1, r2 = random.random(), random.random()
        
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self):
        """Update particle position."""
        self.position += self.velocity
        
        # Apply bounds and minimum weight constraint
        for i in range(self.dimension):
            self.position[i] = max(self.min_weight, min(self.bounds[i][1], self.position[i]))
        
        # Normalize to ensure weights sum to 1
        self.position = self.position / np.sum(self.position)
        
        # Ensure minimum weights are maintained after normalization
        for i in range(self.dimension):
            if self.position[i] < self.min_weight:
                self.position[i] = self.min_weight
        
        # Renormalize after applying minimum weights
        self.position = self.position / np.sum(self.position)
    
    def update_best(self):
        """Update particle's best position if current fitness is better."""
        if self.fitness > self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = self.fitness

class PSO:
    """Particle Swarm Optimization algorithm for portfolio optimization."""
    
    def __init__(self, n_particles: int, n_iterations: int, w: float = 0.9, c1: float = 2.0, c2: float = 2.0):
        """
        Initialize PSO algorithm.
        
        Args:
            n_particles (int): Number of particles
            n_iterations (int): Number of iterations
            w (float): Inertia weight
            c1 (float): Cognitive parameter
            c2 (float): Social parameter
        """
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.particles = []
        self.global_best_position = None
        self.global_best_fitness = float('-inf')
        self.fitness_history = []
    
    def initialize_particles(self, dimension: int, bounds: List[Tuple[float, float]], min_weight: float = 0.01):
        """Initialize particles."""
        self.particles = [Particle(dimension, bounds, min_weight) for _ in range(self.n_particles)]
    
    def optimize(self, fitness_function, returns_data: pd.DataFrame, risk_tolerance: float = 0.5):
        """
        Run PSO optimization.
        
        Args:
            fitness_function: Function to evaluate fitness
            returns_data (pd.DataFrame): Returns data for assets
            risk_tolerance (float): Risk tolerance parameter
            
        Returns:
            Tuple[np.ndarray, float]: Best position and fitness
        """
        dimension = len(returns_data.columns)
        bounds = [(0.0, 1.0) for _ in range(dimension)]
        min_weight = 0.01  # 1% minimum weight per asset
        
        self.initialize_particles(dimension, bounds, min_weight)
        
        log_model_action(None, "PSO Initialization", f"Initialized {self.n_particles} particles for {dimension} assets")
        
        for iteration in range(self.n_iterations):
            # Evaluate fitness for all particles
            for particle in self.particles:
                particle.fitness = fitness_function(particle.position, returns_data, risk_tolerance)
                particle.update_best()
                
                # Update global best
                if particle.fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Update velocities and positions
            for particle in self.particles:
                particle.update_velocity(self.global_best_position, self.w, self.c1, self.c2)
                particle.update_position()
            
            # Record fitness history
            self.fitness_history.append(self.global_best_fitness)
            
            # Log progress
            if iteration % 10 == 0 or iteration == self.n_iterations - 1:
                log_model_action(None, "PSO Progress", f"Iteration {iteration+1}/{self.n_iterations}, Best Fitness: {self.global_best_fitness:.6f}")
        
        return self.global_best_position, self.global_best_fitness

class IS2Module:
    """IS-II module for portfolio optimization using PSO."""
    
    def __init__(self):
        """Initialize the IS-II module."""
        self.logger = setup_logger("IS2Module")
        self.data_fetcher = DataFetcher()
    
    def run(self):
        """Main execution method for the IS-II module."""
        try:
            # Get user inputs
            tickers, pso_params, risk_tolerance, time_period = self._get_user_inputs()
            
            # Fetch historical data
            self.logger.info("Fetching historical data for portfolio optimization")
            returns_data = self._fetch_returns_data(tickers, time_period)
            
            # Run PSO optimization
            self.logger.info("Running PSO portfolio optimization")
            optimal_weights, best_fitness = self._run_pso_optimization(returns_data, pso_params, risk_tolerance)
            
            # Calculate equal allocation portfolio for comparison
            equal_weights = np.ones(len(tickers)) / len(tickers)
            equal_metrics = self._calculate_portfolio_metrics(equal_weights, returns_data)
            
            # Calculate portfolio metrics
            self.logger.info("Calculating portfolio metrics")
            portfolio_metrics = self._calculate_portfolio_metrics(optimal_weights, returns_data)
            
            # Display results
            self._display_results(tickers, optimal_weights, portfolio_metrics, equal_weights, equal_metrics, best_fitness, time_period)
            
        except Exception as e:
            self.logger.error(f"Error in IS-II module: {e}")
            print(f"Error: {e}")
    
    def _get_user_inputs(self) -> Tuple[List[str], Dict, float, int]:
        """Get user inputs for portfolio optimization."""
        print("\n" + "-"*40)
        print("PORTFOLIO OPTIMIZATION INPUTS")
        print("-"*40)
        
        # Get stock tickers
        while True:
            tickers_input = input("Enter stock tickers (comma-separated, e.g., AAPL,MSFT,GOOGL): ").strip()
            if tickers_input:
                tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
                if len(tickers) >= 2:
                    break
                print("Please enter at least 2 tickers.")
            else:
                print("Please enter valid ticker symbols.")
        
        # Get investment time period
        print("\nInvestment Time Period:")
        while True:
            try:
                time_period = int(input("Investment time period in days (30-365, default 365): ") or "365")
                if 30 <= time_period <= 365:
                    break
                print("Please enter a number between 30 and 365.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get PSO parameters
        print("\nPSO Parameters:")
        while True:
            try:
                n_particles = int(input("Number of particles (20-100, default 50): ") or "50")
                if 20 <= n_particles <= 100:
                    break
                print("Please enter a number between 20 and 100.")
            except ValueError:
                print("Please enter a valid number.")
        
        while True:
            try:
                n_iterations = int(input("Number of iterations (50-200, default 100): ") or "100")
                if 50 <= n_iterations <= 200:
                    break
                print("Please enter a number between 50 and 200.")
            except ValueError:
                print("Please enter a valid number.")
        
        # Get risk tolerance
        while True:
            try:
                risk_tolerance = float(input("Risk tolerance (0.1-1.0, default 0.5): ") or "0.5")
                if 0.1 <= risk_tolerance <= 1.0:
                    break
                print("Please enter a number between 0.1 and 1.0.")
            except ValueError:
                print("Please enter a valid number.")
        
        pso_params = {
            'n_particles': n_particles,
            'n_iterations': n_iterations,
            'w': 0.9,  # Inertia weight
            'c1': 2.0,  # Cognitive parameter
            'c2': 2.0   # Social parameter
        }
        
        self.logger.info(f"User inputs - Tickers: {tickers}, Time Period: {time_period} days, Particles: {n_particles}, Iterations: {n_iterations}, Risk: {risk_tolerance}")
        return tickers, pso_params, risk_tolerance, time_period
    
    def _fetch_returns_data(self, tickers: List[str], time_period: int) -> pd.DataFrame:
        """Fetch historical data and calculate returns."""
        log_data_processing(self.logger, "Data Fetch", f"Fetching data for {len(tickers)} assets")
        
        returns_data = {}
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=time_period)).strftime("%Y-%m-%d")
        
        for ticker in tickers:
            try:
                # Fetch stock data
                data = self.data_fetcher.fetch_stock_data(ticker, start_date, end_date)
                
                if data.empty:
                    self.logger.warning(f"No data available for {ticker}")
                    continue
                
                # Calculate returns
                returns = self.data_fetcher.calculate_returns(data)
                returns_data[ticker] = returns
                
                log_data_processing(self.logger, "Returns Calculation", f"Calculated returns for {ticker}: {len(returns)} data points")
                
            except Exception as e:
                self.logger.error(f"Error fetching data for {ticker}: {e}")
                print(f"Warning: Could not fetch data for {ticker}")
        
        if not returns_data:
            raise Exception("No data available for any of the specified tickers")
        
        # Align all returns data
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 30:
            raise Exception("Insufficient data for portfolio optimization. Need at least 30 days of data.")
        
        log_data_processing(self.logger, "Data Alignment", f"Aligned returns data: {len(returns_df)} days, {len(returns_df.columns)} assets")
        
        return returns_df
    
    def _run_pso_optimization(self, returns_data: pd.DataFrame, pso_params: Dict, risk_tolerance: float) -> Tuple[np.ndarray, float]:
        """Run PSO optimization for portfolio weights."""
        log_model_action(self.logger, "PSO Setup", f"Setting up PSO with {pso_params['n_particles']} particles, {pso_params['n_iterations']} iterations")
        
        # Create PSO instance
        pso = PSO(
            n_particles=pso_params['n_particles'],
            n_iterations=pso_params['n_iterations'],
            w=pso_params['w'],
            c1=pso_params['c1'],
            c2=pso_params['c2']
        )
        
        # Define fitness function (Sharpe ratio)
        def fitness_function(weights, returns_data, risk_tolerance):
            return self._calculate_sharpe_ratio(weights, returns_data, risk_tolerance)
        
        # Run optimization
        optimal_weights, best_fitness = pso.optimize(fitness_function, returns_data, risk_tolerance)
        
        log_model_action(self.logger, "PSO Completion", f"Optimization completed. Best fitness: {best_fitness:.6f}")
        
        return optimal_weights, best_fitness
    
    def _calculate_sharpe_ratio(self, weights: np.ndarray, returns_data: pd.DataFrame, risk_tolerance: float) -> float:
        """
        Calculate Sharpe ratio for given portfolio weights.
        
        Args:
            weights (np.ndarray): Portfolio weights
            returns_data (pd.DataFrame): Returns data
            risk_tolerance (float): Risk tolerance parameter
            
        Returns:
            float: Sharpe ratio
        """
        # Calculate portfolio returns
        portfolio_returns = (returns_data * weights).sum(axis=1)
        
        # Calculate expected return and volatility
        expected_return = portfolio_returns.mean() * 252  # Annualized
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate Sharpe ratio (with risk-free rate = 0 for simplicity)
        if volatility == 0:
            return 0
        
        sharpe_ratio = expected_return / volatility
        
        # Apply risk tolerance penalty
        risk_penalty = risk_tolerance * (volatility ** 2)
        
        return sharpe_ratio - risk_penalty
    
    def _calculate_portfolio_metrics(self, weights: np.ndarray, returns_data: pd.DataFrame) -> Dict:
        """Calculate comprehensive portfolio metrics."""
        log_data_processing(self.logger, "Metrics Calculation", "Calculating portfolio performance metrics")
        
        # Calculate portfolio returns
        portfolio_returns = (returns_data * weights).sum(axis=1)
        
        # Basic metrics
        expected_return = portfolio_returns.mean() * 252  # Annualized
        volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
        sharpe_ratio = expected_return / volatility if volatility > 0 else 0
        
        # Individual asset metrics
        asset_metrics = {}
        for i, ticker in enumerate(returns_data.columns):
            asset_returns = returns_data[ticker]
            asset_expected_return = asset_returns.mean() * 252
            asset_volatility = asset_returns.std() * np.sqrt(252)
            
            asset_metrics[ticker] = {
                'weight': weights[i],
                'expected_return': asset_expected_return,
                'volatility': asset_volatility
            }
        
        return {
            'portfolio': {
                'expected_return': expected_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio
            },
            'assets': asset_metrics
        }
    
    def _display_results(self, tickers: List[str], optimal_weights: np.ndarray, portfolio_metrics: Dict, equal_weights: np.ndarray, equal_metrics: Dict, best_fitness: float, time_period: int):
        """Display portfolio optimization results."""
        print("\n" + "="*80)
        print("                    PORTFOLIO OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Investment Time Period: {time_period} days")
        
        # Display optimal portfolio allocation
        print(f"\nOptimal Portfolio Allocation (PSO Optimized):")
        print("-" * 70)
        print(f"{'Ticker':<10} | {'Weight':<10} | {'Expected Return':<15} | {'Volatility':<12}")
        print("-" * 70)
        
        for i, ticker in enumerate(tickers):
            if i < len(optimal_weights):
                weight = optimal_weights[i]
                asset_metrics = portfolio_metrics['assets'].get(ticker, {})
                expected_return = asset_metrics.get('expected_return', 0) * 100
                volatility = asset_metrics.get('volatility', 0) * 100
                
                print(f"{ticker:<10} | {weight:>8.1%} | {expected_return:>13.2f}% | {volatility:>10.2f}%")
        
        # Display equal allocation portfolio for comparison
        print(f"\nEqual Allocation Portfolio (Benchmark):")
        print("-" * 70)
        print(f"{'Ticker':<10} | {'Weight':<10} | {'Expected Return':<15} | {'Volatility':<12}")
        print("-" * 70)
        
        for i, ticker in enumerate(tickers):
            if i < len(equal_weights):
                weight = equal_weights[i]
                asset_metrics = equal_metrics['assets'].get(ticker, {})
                expected_return = asset_metrics.get('expected_return', 0) * 100
                volatility = asset_metrics.get('volatility', 0) * 100
                
                print(f"{ticker:<10} | {weight:>8.1%} | {expected_return:>13.2f}% | {volatility:>10.2f}%")
        
        # Display portfolio comparison
        optimal_portfolio = portfolio_metrics['portfolio']
        equal_portfolio = equal_metrics['portfolio']
        
        print(f"\nPortfolio Comparison:")
        print("-" * 50)
        print(f"{'Metric':<20} | {'PSO Optimized':<15} | {'Equal Allocation':<15}")
        print("-" * 50)
        print(f"{'Expected Return':<20} | {optimal_portfolio['expected_return']*100:>13.2f}% | {equal_portfolio['expected_return']*100:>13.2f}%")
        print(f"{'Volatility':<20} | {optimal_portfolio['volatility']*100:>13.2f}% | {equal_portfolio['volatility']*100:>13.2f}%")
        print(f"{'Sharpe Ratio':<20} | {optimal_portfolio['sharpe_ratio']:>13.4f} | {equal_portfolio['sharpe_ratio']:>13.4f}")
        
        # Calculate improvement
        return_improvement = ((optimal_portfolio['expected_return'] - equal_portfolio['expected_return']) / equal_portfolio['expected_return']) * 100
        sharpe_improvement = ((optimal_portfolio['sharpe_ratio'] - equal_portfolio['sharpe_ratio']) / equal_portfolio['sharpe_ratio']) * 100
        
        print(f"\nImprovement over Equal Allocation:")
        print("-" * 40)
        print(f"Return Improvement: {return_improvement:+.2f}%")
        print(f"Sharpe Improvement: {sharpe_improvement:+.2f}%")
        print(f"Optimization Score: {best_fitness:.6f}")
        
        # Display risk analysis
        print(f"\nRisk Analysis:")
        print("-" * 20)
        
        # Calculate concentration risk
        max_weight = np.max(optimal_weights)
        concentration_risk = "High" if max_weight > 0.4 else "Medium" if max_weight > 0.25 else "Low"
        print(f"Concentration Risk: {concentration_risk} (max weight: {max_weight:.1%})")
        
        # Calculate diversification ratio
        individual_vol = np.mean([asset['volatility'] for asset in portfolio_metrics['assets'].values()])
        portfolio_vol = optimal_portfolio['volatility']
        diversification_ratio = individual_vol / portfolio_vol if portfolio_vol > 0 else 0
        print(f"Diversification Ratio: {diversification_ratio:.2f}")
        
        # Display optimization details
        print(f"\nOptimization Details:")
        print("-" * 25)
        print(f"Algorithm: Particle Swarm Optimization (PSO)")
        print(f"Objective: Maximize Sharpe Ratio with Risk Penalty")
        print(f"Constraint: Weights sum to 100%, minimum 1% per asset")
        print(f"Time Period: {time_period} days")
        
        print("\n" + "="*80)
        print("Note: Portfolio optimization results are for educational purposes only.")
        print("Past performance does not guarantee future results.")
        print("Consider consulting a financial advisor before making investment decisions.")
        print("="*80)
