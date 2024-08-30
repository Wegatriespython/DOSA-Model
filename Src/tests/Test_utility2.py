import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utilities.utility_function import maximize_utility
def main():
    results = maximize_utility(2.5262568012801, 0.0546875, 1.94, 2, 10)
    print(results)
#Calling maximize utility with savings: 2.5262568012801,
#wage: 0.0546875, average consumption good price: 1.9428917972287767,
#price history: 1.9574048384119385, total working hours: 10
if __name__ == '__main__':
    main()
