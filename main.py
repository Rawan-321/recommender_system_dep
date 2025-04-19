from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidfvec = TfidfVectorizer(min_df=2, max_df=0.9)
api_key = "aafe6ff1ce31c14363eabc2b2c327ddf1585266052ddba338769485d5dfbfc2b"

app = FastAPI()


class Request(BaseModel):
    check_in: str
    check_out: str
    user_profile: dict


@app.post("/recommend")
def create_trip(details: Request):
    trip = get_recommendations(details.check_in, details.check_out, details.user_profile)
    if not trip:
        raise HTTPException(status_code=404, detail="trip empty")
    if not details.user_profile:
        raise HTTPException(status_code=404, detail="user profile empty")
    print(trip)
    return trip

# ---- Recommendation ---- #


def get_recommendations(start_date, end_date, user_profile):
    # should return at least one result
    recommendation = {}
    destination = get_destinations_data(user_profile)
    if destination != "":
        hotel_info = get_accommodation_data(q=destination, check_in=start_date, check_out=end_date,
                                            user_profile=user_profile)
        activities_info = get_activities_data(destination, user_profile)

        recommendation["destination"] = destination
        recommendation["accommodation"] = hotel_info
        recommendation["activitie"] = activities_info

        return recommendation
    else:
        return recommendation


# ---- user profile ---- #


def create_user_profile(tfidfvec, df, user_profile):
    padding_values = ""
    max_list_length = 0
    for key in user_profile.keys():
        max_list_length = max(max_list_length, len(user_profile[key]))

    for dict_values in user_profile.keys():
        while len(user_profile[dict_values]) < max_list_length:
            user_profile[dict_values].append(padding_values)

    user_history_and_preference_df = pd.DataFrame(user_profile)

    user_history_and_preference_df["combined_data"] = (user_history_and_preference_df['initial_preferences'] + ' ' +
                                                       user_history_and_preference_df['history'])
    print(user_history_and_preference_df)

    vectorized_data = tfidfvec.transform(user_history_and_preference_df['combined_data'])
    tfidf_df = pd.DataFrame(vectorized_data.toarray(),
                            columns=tfidfvec.get_feature_names_out())
    user_profile = tfidf_df.mean()
    # user_profile_array = user_profile.values.reshape(1, -1)
    # user_profile_df = pd.DataFrame(user_profile_array,
    #                                index=['user'])

    user_profile_similarities = cosine_similarity(user_profile.values.reshape(1, -1), df)
    user_profile_similarities_df = pd.DataFrame(user_profile_similarities.T,
                                                index=df.index,
                                                columns=["similarity_score"])
    sorted_similarities_df = user_profile_similarities_df.sort_values('similarity_score', ascending=False)
    return sorted_similarities_df.head()


# ---- destinations ---- #


def load_destinations_data():
    with open("destinations.json", 'r') as json_file:
        destinations = json.load(json_file)

    df = pd.DataFrame(destinations.items(), columns=['city_name', 'description'])

    return df


def create_destinations_vectors(df):
    vectorized_data = tfidfvec.fit_transform(df['description'])
    tfidf_df = pd.DataFrame(vectorized_data.toarray(),
                            columns=tfidfvec.get_feature_names_out())

    # the data frame rows are: hotel names, the columns are the features names and values are tfidf values
    tfidf_df.index = df['city_name']

    return tfidf_df


def get_destinations_data(user_profile):
    df = load_destinations_data()
    destinations_df = create_destinations_vectors(df=df)
    sorted_destinations_similarities = create_user_profile(tfidfvec, destinations_df, user_profile)
    # print(sorted_destinations_similarities.to_string())
    return sorted_destinations_similarities.index.values[0]


# ---- Accommodations ---- #


def clean_json_data(api_json):
    all_json_data = api_json
    empty_data_frame = pd.DataFrame()
    filtered_properties = []
    df = pd.DataFrame(columns=['name', 'description', 'rating', 'reviews', 'price', 'hotel_class'])

    # from line 41 to line 49 json data is filtered so all contain necessity attributes
    if 'properties' in all_json_data:
        properties_list = all_json_data['properties']

        for prop in properties_list:  # the "eco_certified" is excluded because most properties do not have the value
            if ("description" in prop and "images" in prop and "reviews" in prop and "overall_rating" in prop
                    and "rate_per_night" in prop and "link" in prop and "name" in prop and
                    "hotel_class" in prop):
                filtered_properties.append(prop)
        properties_list = filtered_properties  # this contains the data that will be used to get the recommended
        # accommodation to send to the application

        # from line 69 to 79, loading data from the properties list to dataframe
        for i in range(0, len(properties_list)):
            df.loc[i] = [properties_list[i]['name'],
                         properties_list[i]['description'],
                         str(properties_list[i]['overall_rating']),
                         str(properties_list[i]['reviews']),
                         str(properties_list[i]['rate_per_night']['extracted_lowest']),
                         properties_list[i]['hotel_class']]
        df["combined_data"] = (df['name'] + ' ' + df['description'] + ' rating ' + df['rating'] +
                               ' total number of reviews ' + df['reviews'] + ' price per night ' +
                               df['price'] + ' ' + df['hotel_class'])

        # print(df.to_string())
        return df, properties_list

    else:
        return empty_data_frame


def create_accommodations_vectors(df):
    vectorized_data = tfidfvec.fit_transform(df['combined_data'])
    tfidf_df = pd.DataFrame(vectorized_data.toarray(),
                            columns=tfidfvec.get_feature_names_out())

    # the data frame rows are: hotel names, the columns are the features names and values are tfidf values
    tfidf_df.index = df['name']

    # this contains how similar each hotel to every other hotel in the df
    cosine_similarity_array = cosine_similarity(tfidf_df)
    # wrapping the cosine similarity array in a data frame
    cosine_similarity_df = pd.DataFrame(cosine_similarity_array,
                                        index=tfidf_df.index,
                                        columns=tfidf_df.index)
    return tfidf_df


def get_accommodation_data(q: str, check_in, check_out, user_profile):
    empty_df = pd.DataFrame()
    params = {
        "engine": "google_hotels",
        "q": q,
        "check_in_date": check_in,
        "check_out_date": check_out,
        "adults": "2",
        "currency": "SAR",
        "hl": "en",
        "api_key": api_key
    }

    try:
        results = requests.get("https://serpapi.com/search?engine=google_hotels", params=params)
        results.raise_for_status()

        accommodations = results.json()
        df, complete_accommodations_info = clean_json_data(accommodations)
        if df.empty:
            return empty_df
        accommodations_df = create_accommodations_vectors(df)
        sorted_accommodations_similarities = create_user_profile(tfidfvec, accommodations_df, user_profile)
        hotel_name = sorted_accommodations_similarities.index.values[0]
        hotel_info = df.loc[df['name'] == hotel_name]
        hotel_complete_info = {}
        for i in complete_accommodations_info:
            if i["name"] == hotel_info.values[0][0]:
                hotel_complete_info = i
                break

        hotel_dict = {}

        hotel_dict["name"] = hotel_complete_info["name"]
        hotel_dict["description"] = hotel_complete_info["description"]
        hotel_dict["rating"] = hotel_complete_info["overall_rating"]
        hotel_dict["reviews"] = hotel_complete_info["reviews"]
        hotel_dict["price"] = hotel_complete_info["rate_per_night"]["extracted_lowest"]
        hotel_dict["link"] = hotel_complete_info["link"]
        hotel_dict["image"] = hotel_complete_info["images"][0]["original_image"]
        hotel_dict["check_in"] = check_in
        hotel_dict["check_out"] = check_out
        if "eco_certified" in hotel_complete_info.keys():
            hotel_dict["eco_certified"] = hotel_complete_info["eco_certified"]

        return hotel_dict

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    except requests.exceptions.JSONDecodeError:
        print("Error: Response is not valid JSON.")


# ---- Activities ---- #


def clean_activities_json_data(api_json):
    all_json_data = api_json
    empty_list = []
    filtered_activities = []
    df = pd.DataFrame(columns=['title', 'description', 'rating', 'reviews'])

    # from line 41 to line 49 json data is filtered so all contain necessity attributes
    if 'events_results' in all_json_data:
        activities_list = all_json_data['events_results']

        for activity in activities_list:  # the "eco_certified" is excluded because most properties do not have the value
            if ("description" in activity and "date" in activity and "venue" in activity and "title" in activity
                    and "address" in activity):
                filtered_activities.append(activity)
        activities_list = filtered_activities  # this contains the data that will be used to get the recommended
        # activities to send to the application

        # go over the address

        # from line 69 to 79, loading data from the activities list to dataframe
        for i in range(0, len(activities_list)):
            df.loc[i] = [activities_list[i]['title'],
                         activities_list[i]['description'],
                         str(activities_list[i]['venue']["rating"]),
                         str(activities_list[i]['venue']['reviews'])]
        df["combined_data"] = (df['title'] + ' ' + df['description'] + ' rating ' + df['rating'] +
                               ' total number of reviews ' + df['reviews'])

        return df, activities_list

    else:
        return empty_list


def create_activities_vectors(df):
    vectorized_data = tfidfvec.fit_transform(df['combined_data'])
    tfidf_df = pd.DataFrame(vectorized_data.toarray(),
                            columns=tfidfvec.get_feature_names_out())

    # the data frame rows are: hotel names, the columns are the features names and values are tfidf values
    tfidf_df.index = df['title']

    # this contains how similar each hotel to every other hotel in the df
    cosine_similarity_array = cosine_similarity(tfidf_df)
    # wrapping the cosine similarity array in a data frame
    cosine_similarity_df = pd.DataFrame(cosine_similarity_array,
                                        index=tfidf_df.index,
                                        columns=tfidf_df.index)
    return tfidf_df


def get_activities_data(q: str, user_profile):
    empty_df = pd.DataFrame()
    activities_list =[]
    activities_dict = {}
    params = {
        "engine": "google_events",
        "q": q,
        "api_key": api_key
    }

    try:
        results = requests.get("https://serpapi.com/search?google_events", params=params)
        results.raise_for_status()
        activities = results.json()
        df, complete_activities_info = clean_activities_json_data(activities)
        if df.empty:
            return empty_df
        activities_df = create_activities_vectors(df)
        sorted_activities_similarities = create_user_profile(tfidfvec, activities_df, user_profile)
        for i in range(len(sorted_activities_similarities)):
            activity_title = sorted_activities_similarities.index.values[i]
            activity_info = df.loc[df['title'] == activity_title]
            activity_complete_info = {}
            for j in complete_activities_info:
                if j["title"] == activity_info.values[0][0]:
                    activity_complete_info = j
                    break
            activities_dict[f'activity {i + 1}'] = {
                'title': activity_info.values[0][0],
                'description': activity_info.values[0][1],
                'rating': activity_info.values[0][2],
                'reviews': activity_info.values[0][3],
                'address': activity_complete_info["address"],
                'date': activity_complete_info["date"]["when"]
            }
            activities_list.append(activities_dict[f'activity {i + 1}'])

        return activities_list

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")

    except requests.exceptions.JSONDecodeError:
        print("Error: Response is not valid JSON.")