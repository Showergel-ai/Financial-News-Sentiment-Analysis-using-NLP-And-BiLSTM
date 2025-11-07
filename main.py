#!/usr/bin/env python3
"""
Financial Project Application - Main Entry Point
Terminal-based application with three modules: DL, NLP, and IS-II
"""

import sys
import argparse
from modules.dl_module import DeepLearningModule
from modules.nlp_module import NLPModule
from modules.is2_module import IS2Module
from utils.logger import setup_logger

def main():
    """Main entry point for the financial project application."""
    parser = argparse.ArgumentParser(description='Financial Project Application')
    parser.add_argument('--module', '-m', 
                       choices=['dl', 'nlp', 'is2', 'menu'],
                       default='menu',
                       help='Module to run: dl (Deep Learning), nlp (NLP), is2 (IS-II), or menu (interactive)')
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger()
    logger.info("Financial Project Application started")
    
    if args.module == 'menu':
        show_menu()
    elif args.module == 'dl':
        run_dl_module()
    elif args.module == 'nlp':
        run_nlp_module()
    elif args.module == 'is2':
        run_is2_module()

def show_menu():
    """Display interactive menu for module selection."""
    print("\n" + "="*60)
    print("           FINANCIAL PROJECT APPLICATION")
    print("="*60)
    print("1. Deep Learning Module - Stock Price Prediction")
    print("2. NLP Module - Financial News Sentiment Analysis")
    print("3. IS-II Module - Portfolio Optimization (PSO)")
    print("4. Exit")
    print("="*60)
    
    while True:
        try:
            choice = input("\nSelect a module (1-4): ").strip()
            
            if choice == '1':
                run_dl_module()
                break
            elif choice == '2':
                run_nlp_module()
                break
            elif choice == '3':
                run_is2_module()
                break
            elif choice == '4':
                print("Exiting application. Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-4.")
        except KeyboardInterrupt:
            print("\nExiting application. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def run_dl_module():
    """Run the Deep Learning module."""
    print("\n" + "="*50)
    print("    DEEP LEARNING MODULE - STOCK PREDICTION")
    print("="*50)
    
    try:
        dl_module = DeepLearningModule()
        dl_module.run()
    except Exception as e:
        print(f"Error running DL module: {e}")

def run_nlp_module():
    """Run the NLP module."""
    print("\n" + "="*50)
    print("    NLP MODULE - SENTIMENT ANALYSIS")
    print("="*50)
    
    try:
        nlp_module = NLPModule()
        nlp_module.run()
    except Exception as e:
        print(f"Error running NLP module: {e}")

def run_is2_module():
    """Run the IS-II module."""
    print("\n" + "="*50)
    print("    IS-II MODULE - PORTFOLIO OPTIMIZATION")
    print("="*50)
    
    try:
        is2_module = IS2Module()
        is2_module.run()
    except Exception as e:
        print(f"Error running IS-II module: {e}")

if __name__ == "__main__":
    main()



