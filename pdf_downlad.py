import os
import requests
import httplib2
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
import concurrent.futures

from bs4 import BeautifulSoup
import tabula
import camelot as cam
output_dir = ".\Outputs"
url = "https://maya.tase.co.il/en/reports/finance"  # remark, headless scraping from maya webpage works only with firefox browser
# links = []
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36",
           "Accept-Language": "en-US,en;q=0.9",
           "Referer": "http://google.com",
           "DNT":"1"
          }

class Crawler:
    def __init__(self):
        self._headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.157 Safari/537.36",
           "Accept-Language": "en-US,en;q=0.9",
           "Referer": "http://google.com",
           "DNT":"1"
          }
        self.options = Options()
        self._service = Service()
        self.options.headless = True
        self.options.add_argument('--disable-blink-features=AutomationControlled')
        self.options.add_argument(f'user-agent={self._headers["User-Agent"]}')
        self.options.add_argument("--window-size=1920x1080")
        self.options.add_argument('--ignore-certificate-errors')
        self.options.add_argument('--allow-running-insecure-content')
        self.options.add_argument("--disable-extensions")
        self.options.add_argument("--proxy-server='direct://'")
        self.options.add_argument("--proxy-bypass-list=*")
        self.options.add_argument("--headless")
        self.options.add_argument("--start-maximized")
        self.options.add_argument('--disable-gpu')
        self.options.add_argument('--disable-dev-shm-usage')
        self.options.add_argument('--no-sandbox')
        self._driver = webdriver.Firefox(service=self._service, options=self.options)
        self._driver.get(url)
        self._driver.get_screenshot_as_file("test5.png")
        self._company_pdf_links = None # dict of key: company name, val: the links for the pdf files
        # print(self._driver.title)


    #TODO rn headless (so it wont open the borwser)


    def get_links(self, company_pdf_links, links):
        browser = self._driver
        elem_list = browser.find_element(By.CSS_SELECTOR, " div.min-height.ng-scope > div > div > div > div > div")
        items = elem_list.find_elements(By.XPATH, '//div[@role="row"]')[1:]
        for item in items:
            title = item.find_element(By.TAG_NAME, 'a').text
            links = [i.get_attribute('href') for i in item.find_elements(By.TAG_NAME, 'a') if
                     "/2/" in i.get_attribute('href')]
            company_pdf_links[title] = set(links)

    def get_pdf_links(self, companies_data_links, pdf_links_list):
        """
        extract and filter only the pdf links of the company from the data link
        :param companies_data_links: the urls for the companies data
        :param pdf_links_list: a pointer to a list which we save the pdf files links
        :return: nothing, only update the given pdf_links_list by using its pointer
        """
        # browser = webdriver.Chrome()
        pdf_links = []
        counter = 0
        while counter < len(companies_data_links):
            self._driver.get(companies_data_links[counter])
            elem = self._driver.find_elements(By.TAG_NAME, 'a')
            unfilterd_links = [i.get_attribute('href')for i in elem]
            filtered_links = list(filter(lambda link: link is not None and
                                         len(link) != 0
                                         and link.endswith('.pdf'), unfilterd_links))

            pdf_links_list += filtered_links
            counter += 1

        # browser.quit()

    def download_pdf(self, pdf_urls):
        """
        download the pdf file from a given url
        :param pdf_urls: the url of pdf to download
        """
        for url in pdf_urls:
            response = requests.get(url)
            if response.status_code == 200:  # 200 is good, 404 is bad
                file_path = os.path.join(output_dir, os.path.basename(url))
                # join the output_dir name with the tail of the url which is the pdf file name
                with open(file_path, 'wb') as f:
                    f.write(response.content)

        # TODO
        # if theres already file in the output dir it will send them as well, need to fix it

    def init(self, company_name = None):
        """
        run the program to find and downland the finances files of all or several companies
        using multi threading
        :param company_name: None by default - download all files, else a list of companies name

        """
        pdf_urls = []
        # company_pdf_links = self.get_data_links()
        print("111111")
        company_pdf_links, links = {}, []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.get_links, company_pdf_links, links)
            executor.submit(self.get_links, company_pdf_links, links)
        self._company_pdf_links = company_pdf_links
        print("company_pdf_links are: \n", self._company_pdf_links)
        counter = 0
        for company in company_pdf_links:
            if company_name is None and counter<4:
                # print(f"company name is: {company} \n pdf_web_links are: {company_pdf_links[company]}")
                self.get_pdf_links(list(company_pdf_links[company]), pdf_urls)
                counter += 1
            if company_name is not None and company in company_name and counter<8:
                self.get_pdf_links(list(company_pdf_links[company]), pdf_urls)
                counter += 1
        print(pdf_urls)
        self.download_pdf(pdf_urls)



    #TODO
    # dived "all" and "some" option into 2 functions. the intit is the "runner" and need the
    # "get_pdf_links" part need to dived to the above options. more over try to make it more time efficient using
    # multi threading
    #     pdf_link_lst = []
    #     print("22222")
    #     print(self._company_pdf_links)
    #     with concurrent.futures.ThreadPoolExecutor() as executor:
    #         executor.map(self.get_pdf_links, list(self._company_pdf_links), pdf_link_lst)
    #         executor.submit(self.get_pdf_links, list(self._company_pdf_links), pdf_link_lst)
    #
    #     print("links are : \n", pdf_link_lst)



if __name__ == '__main__':
    crawl = Crawler()
    crawl.init()
    # crawl.pull_data()
    # runner(["ויתניה", "אנלייבקס", "tool"])








