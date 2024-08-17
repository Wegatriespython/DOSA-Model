import unittest
from mesa_market_matching import market_matching

class TestMarketMatching(unittest.TestCase):

    def test_normal_market_conditions(self):
        buyers = [
            (5, 10, 'B1'),  # Quantity, Max Price, ID
            (3, 8, 'B2'),
            (4, 6, 'B3'),
            (6, 4, 'B4')
        ]
        sellers = [
            (4, 2, 'S1'),  # Quantity, Min Price, ID
            (5, 4, 'S2'),
            (3, 6, 'S3'),
            (6, 8, 'S4')
        ]
        result = market_matching(buyers, sellers)
        print("bbbbtest 1",result)
        self.assertEqual(len(result), 4)
        self.assertAlmostEqual(sum(trade[2] for trade in result), 12)
        self.assertTrue(all(6 <= trade[3] <= 9 for trade in result))

    def test_excess_demand(self):
        buyers = [
            (10, 12, 'B1'),
            (8, 10, 'B2'),
            (6, 8, 'B3')
        ]
        sellers = [
            (5, 6, 'S1'),
            (7, 7, 'S2'),
            (3, 9, 'S3')
        ]
        result = market_matching(buyers, sellers)
        print("ssstest2", result)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(sum(trade[2] for trade in result), 15)
        self.assertTrue(all(9 <= trade[3] <= 10.5 for trade in result))

    def test_excess_supply(self):
        buyers = [
            (3, 8, 'B1'),
            (4, 7, 'B2'),
            (2, 6, 'B3')
        ]
        sellers = [
            (5, 4, 'S1'),
            (6, 5, 'S2'),
            (8, 6, 'S3'),
            (4, 7, 'S4')
        ]
        result = market_matching(buyers, sellers)
        print("dddtest3", result)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(sum(trade[2] for trade in result), 9)
        self.assertTrue(all(5.5 <= trade[3] <= 7.5 for trade in result))

    def test_edge_cases(self):
        # No matches possible
        buyers = [(5, 5, 'B1')]
        sellers = [(5, 6, 'S1')]

        result = market_matching(buyers, sellers)
        print("anothertest7",result)
        self.assertEqual(len(result), 0)

        # Single possible trade
        buyers = [(5, 10, 'B1')]
        sellers = [(5, 10, 'S1')]
        result = market_matching(buyers, sellers)
        print("ggggtest4", result)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][2], 5)
        self.assertEqual(result[0][3], 10)

        # Multiple buyers, single seller
        buyers = [(2, 12, 'B1'), (3, 10, 'B2'), (4, 8, 'B3')]
        sellers = [(10, 7, 'S1')]

        result = market_matching(buyers, sellers)
        print("kkkkTest5", result)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(sum(trade[2] for trade in result), 9)
        self.assertTrue(all(9.5 <= trade[3] <= 10 for trade in result))

if __name__ == '__main__':
    unittest.main()
