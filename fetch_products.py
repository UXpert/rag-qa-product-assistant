import requests
import csv

# Replace with your chosen product API endpoint
API_URL = "https://fakestoreapi.com/products"

def fetch_and_save_products():
    response = requests.get(API_URL)
    products = response.json()

    # Save product data to CSV
    with open("products.csv", mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "title", "description", "price", "category"])

        for product in products:
            writer.writerow([
                product["id"],
                product["title"],
                product["description"],
                product["price"],
                product["category"]
            ])
    print("✅ Products fetched and saved to products.csv")

if __name__ == "__main__":
    fetch_and_save_products()