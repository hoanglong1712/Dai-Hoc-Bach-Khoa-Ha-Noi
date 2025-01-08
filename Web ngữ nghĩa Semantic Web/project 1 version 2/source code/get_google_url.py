import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

def get_url(keyword):
    """
    search for the given keyword on goole and return the first result url
    :param keyword: the given keyword
    :return: the url of the first result record or None if no result is found
    """

    # set driver webdriver  using Chrome
    driver = webdriver.Chrome()
    result_url = None
    try:
        # navigate to google
        driver.get('https://www.google.com')
        time.sleep(5)
        # find the search bar and enter the keyword

        search_bar = driver.find_element('name', 'q')
        search_bar.send_keys(keyword)
        time.sleep(5)
        search_bar.send_keys(Keys.RETURN)

        first_result = driver.find_element('xpath',
                                           "//div[@class='yuRUbf']/div/span/a")
        print(first_result)
        result_url = first_result.get_attribute('href')

        time.sleep(5)
        pass
    except Exception as e:
        print(f'An error occurred {e}')
        result_url = None
        pass
    finally:
        driver.quit()
        pass

    return result_url