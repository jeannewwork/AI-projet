import pandas as pd
import numpy as np


TARGET = "IsHAB_AIDE_SOC"


def is_missing(value):
    return pd.isna(value)


def bool_value(row, col):
    if col not in row or is_missing(row[col]):
        return False
    return bool(row[col])


def add_if_present(parts, sentence):
    if sentence is not None and sentence != "":
        parts.append(sentence)


def describe_main_identity(row):
    parts = []

    type_principal = row.get("type_principal", None)
    legal_status = row.get("legal_status", None)

    city = row.get("coordinates.city", None)
    dept = row.get("coordinates.deptname", None)

    if not is_missing(type_principal):
        sentence = f"This establishment is a {type_principal}"
    else:
        sentence = "This establishment has an unknown main type"

    location = []

    if not is_missing(city):
        location.append(str(city))

    if not is_missing(dept):
        location.append(str(dept))

    if location:
        sentence += " located in " + ", ".join(location)

    sentence += "."

    parts.append(sentence)

    if not is_missing(legal_status):
        parts.append(f"Its legal status is {legal_status}.")

    return parts


def describe_capacity(row):
    capacity = row.get("capacity", None)

    if is_missing(capacity):
        return None

    return f"It has a capacity of {int(capacity)} places."


def describe_establishment_types(row):
    mapping = {
        "IsEHPAD": "EHPAD",
        "IsEHPA": "EHPA",
        "IsESLD": "ESLD",
        "IsRA": "residence autonomy",
        "IsAJA": "day care facility",
    }

    types = []

    for col, label in mapping.items():
        if bool_value(row, col):
            types.append(label)

    if not types:
        return None

    return "It is classified as: " + ", ".join(types) + "."


def describe_services(row):
    mapping = {
        "IsHCOMPL": "complete accommodation",
        "IsHTEMPO": "temporary accommodation",
        "IsACC_JOUR": "day care",
        "IsACC_NUIT": "night care",
        "IsALZH": "an Alzheimer specialized unit",
        "IsUHR": "a reinforced accommodation unit",
        "IsPASA": "an adapted care and activity center",
        "IsPUV": "a small living unit",
    }

    services = []

    for col, label in mapping.items():
        if bool_value(row, col):
            services.append(label)

    if not services:
        return "No specialized service is indicated."

    return "It offers " + ", ".join(services) + "."


def describe_housing_support(row):
    if bool_value(row, "IsCONV_APL") or row.get("aide_logement_apl", 0) == 1:
        return "It has APL housing support."

    return "It does not have APL housing support."


def describe_housing_types(row):
    mapping = {
        "IsF1": "F1 housing",
        "IsF1Bis": "F1Bis housing",
        "IsF2": "F2 housing",
    }

    housing_types = []

    for col, label in mapping.items():
        if bool_value(row, col):
            housing_types.append(label)

    if not housing_types:
        return None

    return "It offers " + ", ".join(housing_types) + "."


def describe_prices(row):
    parts = []

    prix_min = row.get("prixMin", None)
    prix_median_dept = row.get("prix_median_dept", None)
    ecart_prix_dept = row.get("ecart_prix_dept", None)
    ratio_prix_dept = row.get("ratio_prix_dept", None)

    if not is_missing(prix_min):
        parts.append(f"Its minimum price is {round(float(prix_min), 2)} euros.")

    if not is_missing(prix_median_dept):
        parts.append(
            f"The median minimum price in its department is {round(float(prix_median_dept), 2)} euros."
        )

    if not is_missing(ecart_prix_dept):
        ecart = float(ecart_prix_dept)

        if ecart < 0:
            parts.append("Its price is below the department median price.")
        elif ecart > 0:
            parts.append("Its price is above the department median price.")
        else:
            parts.append("Its price is equal to the department median price.")

    if not is_missing(ratio_prix_dept):
        ratio = float(ratio_prix_dept)

        if ratio < 0.8:
            parts.append("Its price is much lower than the department median.")
        elif ratio < 1:
            parts.append("Its price is slightly lower than the department median.")
        elif ratio <= 1.2:
            parts.append("Its price is close to the department median.")
        else:
            parts.append("Its price is much higher than the department median.")

    return parts


def describe_price_completeness(row):
    parts = []

    nb_ehpad = row.get("nb_prix_ehpad_renseignes", None)
    nb_ra = row.get("nb_prix_ra_renseignes", None)

    if not is_missing(nb_ehpad):
        parts.append(f"It has {int(nb_ehpad)} EHPAD price fields filled.")

    if not is_missing(nb_ra):
        parts.append(f"It has {int(nb_ra)} residence autonomy price fields filled.")

    return parts


def row_to_text(row):
    parts = []

    parts.extend(describe_main_identity(row))

    add_if_present(parts, describe_establishment_types(row))
    add_if_present(parts, describe_capacity(row))
    add_if_present(parts, describe_services(row))
    add_if_present(parts, describe_housing_support(row))
    add_if_present(parts, describe_housing_types(row))

    parts.extend(describe_prices(row))
    parts.extend(describe_price_completeness(row))

    return " ".join(parts)


def dataframe_to_texts(df):
    return df.apply(row_to_text, axis=1).tolist()


def save_text_dataset(df, output_path, target_col=TARGET):
    texts = dataframe_to_texts(df)

    output_df = pd.DataFrame({
        "text": texts
    })

    if target_col in df.columns:
        output_df[target_col] = df[target_col].values

    output_df.to_csv(output_path, index=False)

    return output_df

df = pd.read_csv("/Users/jeannegautier/Projets/AI-projet/data/etablissements_clean_supervised.csv")

text_df = save_text_dataset(
    df,
    "/Users/jeannegautier/Projets/AI-projet/data/etablissements_texts.csv"
)

print(text_df.head())
print(text_df["text"].iloc[0])