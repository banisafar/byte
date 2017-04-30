import unittest
import pandas as pd
from bdc import make_a_feature_list, solve_MSE, data, test_data, OLS_fit_predict, small_data, small_test, prices, zipcode_data, all_data
from pandas.util.testing import assert_frame_equal

sample=pd.DataFrame({'A' : [0,0],   # use this for solve_MSE in test4
                    'B' : [1,2]})


class BDC_Test(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def test1(self):
        self.assertTrue(len(data)>len(test_data), msg = 'didnt divide data up correctly')  # make sure that we split data split correct

    def test2(self):
        self.assertNotIsInstance(make_a_feature_list(data,'price'),list, msg= 'check test 2')  #isnt a list, keeps saying pandas is undefined (why?)

    def test3(self):
        self.assertEqual(len(make_a_feature_list(data,'price')),len(data), msg= 'check test 3')

    def test4(self):
        self.assertEqual(solve_MSE(sample['A'],sample['B']), 2.5, msg='MSE solved incorrectly')

    def test5(self):
        self.assertNotIsInstance(OLS_fit_predict(prices,small_data,small_test),list,msg='check test 5') # ask lesley upload pandas as type

    def test6(self):
        self.assertTrue(len(OLS_fit_predict(prices,small_data,small_test)) == len(test_data), msg = 'these lengths should be equal')
        self.assertFalse(len(OLS_fit_predict(prices,small_data,small_test)) == len(data), msg = ' if ran the data on the traiing data')
        
    def test7(self):
        self.assertEqual(len(data['zipcode'].unique()) , len(all_data['zipcode'].unique()) , msg='make sure data split where all zips represented, if not - change seed')
        
    def test8(self):
       self.assertEqual(zipcode_data[zipcode_data['price'] == max(zipcode_data['price'])].index,98039, msg='most expensive is 98039')
       self.assertEqual(zipcode_data[zipcode_data['price'] == min(zipcode_data['price'])].index,98168, msg='least expensive is 98168')       
if __name__ == '__main__':
    unittest.main()


##def test5(self):
##        self.assertNotEqual(make_a_feature_list(data,'price'), data['price'], msg='these shouldnt be equal since price wont have a constant ') 
