import pickle

cv = pickle.load(open("models/cv.pkl", 'rb')) # rb --> read byte, afou einai se binary morfi
clf = pickle.load(open("models/clf.pkl", 'rb')) # models/clf.pkl <-- dino to sinexes monopati
# etsi metafero ta antikimena apo to notebook gia melontiki xrisi

def model_predict(email):
    if email == "":
        return ""
    tokenized_email = cv.transform([email]) # kani tokenized dld spai to keimeno token kai kathe 
                                            # token efarmozete to multi-hot 
    prediction = clf.predict(tokenized_email) # kai kani tin problepsi

    # If the email is spam prediction should be 1
    prediction = 1 if prediction == 1 else -1 
    return prediction