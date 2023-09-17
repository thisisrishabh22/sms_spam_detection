from sms_spam_predict import predict_spam


result = ["The following SMS is a SPAM", "The following SMS is normal"]


sample_sms = [
    "Hi! You are pre-qulified for Premium SBI Credit Card. Also get Rs.500 worth Amazon Gift Card*, 10X Rewards Point* & more. Click ",
    "[Update] Congratulations Nile Yogesh, You account is activated for investment in Stocks. Click to invest now",
    "Your Stock broker FALANA BROKING LIMITED reported your fund balance Rs.1500.5 & securities balance 0.0 as ",
    "We noticed some unusual activity on your bank card. Please reactivate your account here [link] Your Amazon account has been suspended.",
    "This is an urgent request to transfer INR 2000 to the Anti Corruption Organisation otherwise your account will be ceased. Click the following link to complete the payment orelse you will regret",
    "Our records show you overpaid for (a product or service). Kindly supply your bank routing and account number to receive your refund.",
    "Your niece has been arrested and needs $7,500. Kindly complete the payment to avoid further lawsuits"
]

for msg in sample_sms:
    if predict_spam(msg):
        print(result[0])
    else:
        print(result[1])
