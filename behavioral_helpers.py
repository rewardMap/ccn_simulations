import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)


def string_to_ev(part):
    ev = 0
    if "null" in part:
        ev = 0
        mod = "null"
    else:
        mod, val = part.split("-")
        val_mod = 1 if mod == "save" else 0.5
        ev = float(val) * val_mod

    return ev, mod



def add_additional_columns(df, new_col_name = [], new_col_value = []):
    for ncn, ncv in zip(new_col_name, new_col_value):
        df[ncn] = ncv

    return df



def process_responses(data, query_filter, group_by_cols, value_col):
    return (
        data
        .query(query_filter)
        .groupby(group_by_cols)
        .size()
        .reset_index(name=value_col)
    )


def calculate_mean_response(data, query_filter, field):
    return data.query(query_filter)[field].mean()


def add_risk_sensitive_meaning(data, response_left=0, response_right=1):
    response_events = data.eval("event_type == 'response'")
    data["save_or_correct"] = np.nan
    data["correct_response"] = np.nan
    data["trial_classification"] = ""

    save_answers = []
    trial_categories = []

    for ii in data.loc[response_events, "trial_type"].values:
        tmp1, tmp2 = ii.split("_")

        trial_category, save_answer = None, None
        if "none" in tmp1 or "none" in tmp2:
            trial_category = "forced"
            save_answer = response_right if "none" in tmp1 else response_left
        elif ii == "risky-80_save-20":
            trial_category = "risky"
            save_answer = response_right
        elif ii == "save-20_risky-80":
            trial_category = "risky"
            save_answer = response_left
        else:
            ev1, m1 = string_to_ev(tmp1)
            ev2, m2 = string_to_ev(tmp2)

            if ev1 == ev2:
                trial_category = "risky"
                save_answer = response_left if m1 == "save" else response_right
            else:
                trial_category = "test"
                save_answer = response_left if ev1 > ev2 else response_right

        save_answers.append(save_answer)
        trial_categories.append(trial_category)

    data.loc[response_events, "save_or_correct"] = np.array(save_answers)
    data.loc[response_events, "trial_classification"] = np.array(trial_categories)

    data["trial_classification"] = data["trial_classification"].replace("", np.nan)
    data["trial_classification"] = (
        data.groupby("trial")["trial_classification"]
        .apply(lambda group: group.ffill().bfill())
        .infer_objects(copy=False)
        .values
    )

    data.loc[response_events, "correct_response"] = (
        risk_data.loc[response_events, "response_button"].astype(float)
        == risk_data.loc[response_events, "save_or_correct"].astype(float)
    ) * 1.0

    return data


def add_twostep_meaning(data):
    matching_dict = {
        "1-2": "expected",
        "0-1": "expected",
        "1-1": "unexpected",
        "0-2": "unexpected",
    }
    reward_meaning = {0: "no-reward", 1: "reward"}

    for fill_col in [
        "stay",
        "transition",
        "transition_reward",
        "rewarded",
        "previous_rewarded",
        "response_order",
    ]:
        data[fill_col] = ""

    stage1 = data.eval("event_type=='stage-1-selection'")
    
    first_response = data.query("event_type=='response'").action.values[::2].astype(int)
    second_response = (
        data.query("event_type=='response'").action.values[1::2].astype(int)
    )

    reward = data.query("event_type == 'trial-end'").reward.values.astype(int)
    transition = data.query(
        "event_type=='stage-2-selection'"
    ).current_location.values.astype(int)
    transition = [
        matching_dict[i + "-" + j]
        for i, j in zip(first_response.astype(str), transition.astype(str))
    ]
    stay = [np.nan] + [i == j for i, j in zip(first_response[:-1], first_response[1:])]
    print('stay', np.mean(stay))
    transition_reward = [
        reward_meaning[r] + "_" + j for r, j in zip(reward, transition)
    ]
    rewarded = [reward_meaning[r] for r in reward]

    response_order = [f'{ii}-{jj}-{kk}' for ii, jj, kk in zip(first_response, transition, second_response)]

    data.loc[stage1, "stay"] = stay
    data.loc[stage1, "transition"] = transition
    data.loc[stage1, "transition_reward"] = transition_reward
    data.loc[stage1, "rewarded"] = rewarded
    data.loc[stage1, "previous_rewarded"] = [np.nan] + rewarded[1:]
    data.loc[stage1, "response_order"] = response_order
    
    for fill_col in [
        "stay",
        "transition",
        "transition_reward",
        "rewarded",
        "previous_rewarded",
        "response_order"
    ]:
        data[fill_col] = data[fill_col].replace("", np.nan)

        data[fill_col] = (
            data.groupby("trial")[fill_col]
            .apply(lambda group: group.ffill().bfill())
            .infer_objects(copy=False)
            .values
        )

    return data



def add_risk_sensitive_meaning(data, response_left=0, response_right=1):
    response_events = data.eval("event_type == 'response'")
    data["save_or_correct"] = np.nan
    data["correct_response"] = np.nan
    data["trial_classification"] = ""

    save_answers = []
    trial_categories = []

    for ii in data.loc[response_events, "trial_type"].values:
        tmp1, tmp2 = ii.split("_")

        trial_category, save_answer = None, None
        if "none" in tmp1 or "none" in tmp2:
            trial_category = "forced"
            save_answer = response_right if "none" in tmp1 else response_left
        elif ii == "risky-80_save-20":
            trial_category = "risky"
            save_answer = response_right
        elif ii == "save-20_risky-80":
            trial_category = "risky"
            save_answer = response_left
        else:
            ev1, m1 = string_to_ev(tmp1)
            ev2, m2 = string_to_ev(tmp2)

            if ev1 == ev2:
                trial_category = "risky"
                save_answer = response_left if m1 == "save" else response_right
            else:
                trial_category = "test"
                save_answer = response_left if ev1 > ev2 else response_right

        save_answers.append(save_answer)
        trial_categories.append(trial_category)

    data.loc[response_events, "save_or_correct"] = np.array(save_answers)
    data.loc[response_events, "trial_classification"] = np.array(trial_categories)

    data["trial_classification"] = data["trial_classification"].replace("", np.nan)
    data["trial_classification"] = (
        data.groupby("trial")["trial_classification"]
        .apply(lambda group: group.ffill().bfill())
        .infer_objects(copy=False)
        .values
    )

    data.loc[response_events, "correct_response"] = (
        data.loc[response_events, "response_button"].astype(float)
        == data.loc[response_events, "save_or_correct"].astype(float)
    ) * 1.0

    return data


def add_twostep_meaning(data):
    matching_dict = {
        "1-2": "expected",
        "0-1": "expected",
        "1-1": "unexpected",
        "0-2": "unexpected",
    }
    reward_meaning = {0: "no-reward", 1: "reward"}

    for fill_col in [
        "stay",
        "transition",
        "transition_reward",
        "rewarded",
        "previous_rewarded",
        "response_order",
    ]:
        data[fill_col] = ""

    stage1 = data.eval("event_type=='stage-1-selection'")
    
    first_response = data.query("event_type=='response'").action.values[::2].astype(int)
    second_response = (
        data.query("event_type=='response'").action.values[1::2].astype(int)
    )

    reward = data.query("event_type == 'trial-end'").reward.values.astype(int)
    transition = data.query(
        "event_type=='stage-2-selection'"
    ).current_location.values.astype(int)
    transition = [
        matching_dict[i + "-" + j]
        for i, j in zip(first_response.astype(str), transition.astype(str))
    ]
    stay = [np.nan] + [i == j for i, j in zip(first_response[:-1], first_response[1:])]
    transition_reward = [
        reward_meaning[r] + "_" + j for r, j in zip(reward, transition)
    ]
    rewarded = [reward_meaning[r] for r in reward]

    response_order = [f'{ii}-{jj}-{kk}' for ii, jj, kk in zip(first_response, transition, second_response)]

    data.loc[stage1, "stay"] = stay
    data.loc[stage1, "stay"] = data.loc[stage1, "stay"].astype(float)
    data.loc[stage1, "transition"] = transition
    data.loc[stage1, "transition_reward"] = transition_reward
    data.loc[stage1, "rewarded"] = rewarded
    data.loc[stage1, "previous_rewarded"] = [np.nan] + rewarded[1:]
    data.loc[stage1, "response_order"] = response_order
    
    for fill_col in [
        "stay",
        "transition",
        "transition_reward",
        "rewarded",
        "previous_rewarded",
        "response_order"
    ]:
        data[fill_col] = data[fill_col].replace("", np.nan)

        data[fill_col] = (
            data.groupby("trial")[fill_col]
            .apply(lambda group: group.ffill().bfill())
            .infer_objects(copy=True)
            .values
        )

    return data


def summary_df_risk_sensitive(risk_data, participant):
    # Extracting mean correct responses (proportions)
    correct_responses = (
        risk_data.query("event_type=='response'")
        .groupby("trial_classification")
        .correct_response.mean()
        .reset_index()
        .rename(columns={"correct_response": "value"})
    )

    total_reward = risk_data.query("event_type == 'trial-end'").total_reward.values[-1]

    count_responses = add_additional_columns(
        correct_responses, ["metric"], ["proportion"]
    )

    # Extracting counts of correct responses
    count_responses = process_responses(
        data=risk_data,
        query_filter="event_type=='response'",
        group_by_cols=["trial_classification", "correct_response"],
        value_col="value",
    )
    count_responses = add_additional_columns(count_responses, ["metric"], ["count"])

    # Extracting non-EV and EV risky proportions and counts
    non_ev_risky_counts = process_responses(
        data=risk_data,
        query_filter="event_type=='response' and trial_type in ['risky-80_save-20', 'save-20_risky-80'] and trial_classification == 'risky'",
        group_by_cols=["correct_response"],
        value_col="value",
    )

    non_ev_risky_counts = add_additional_columns(
        non_ev_risky_counts, ["metric", "trial_classification"], ["count", "non-ev"]
    )

    non_ev_risky_mean = calculate_mean_response(
        data=risk_data,
        query_filter=" event_type=='response' and trial_type in ['risky-80_save-20', 'save-20_risky-80'] and trial_classification == 'risky'",
        field='correct_response'
    )

    ev_risky_counts = process_responses(
        data=risk_data,
        query_filter="event_type=='response' and trial_type not in ['risky-80_save-20', 'save-20_risky-80'] and trial_classification == 'risky'",
        group_by_cols=["correct_response"],
        value_col="value",
    )
    ev_risky_counts = add_additional_columns(
        ev_risky_counts, ["metric", "trial_classification"], ["count", "ev"]
    )

    ev_risky_mean = calculate_mean_response(
        data=risk_data,
        query_filter="event_type=='response' and trial_type not in ['risky-80_save-20', 'save-20_risky-80'] and trial_classification == 'risky'",
        field='correct_response'
    )

    # Combine risky data (proportions)
    risky_sep_probs = pd.DataFrame(
        {
            "trial_classification": ["non-ev", "ev"],
            "value": [non_ev_risky_mean, ev_risky_mean],
            "metric": "proportion",
        }
    )

    # Combine everything into a long-format dataframe
    long_format_df = pd.concat(
        [
            correct_responses.assign(
                participant=participant, total_reward=total_reward
            ),
            count_responses.assign(participant=participant, total_reward=total_reward),
            non_ev_risky_counts.assign(
                participant=participant, total_reward=total_reward
            ),
            ev_risky_counts.assign(participant=participant, total_reward=total_reward),
            risky_sep_probs.assign(participant=participant, total_reward=total_reward),
        ],
        ignore_index=True,
    )

    return long_format_df



def summary_df_gonogo(gonogo_data_pre, participant_id):

    total_reward = gonogo_data_pre.query("event_type == 'trial-end'").total_reward.values[-1]
    
    
    nogos = (
        gonogo_data_pre.query('event_type=="response" or event_type=="response-time-out"')
        .groupby("trial_type").action
        .mean()
    )
    nogos['go-punish'] = 1 - nogos['go-punish']
    nogos['go-win'] = 1 - nogos['go-win']
    
    nogos = pd.DataFrame({'trial_type': list(nogos.index) + ['total'], 'value': nogos.values.tolist() + [nogos.mean()]})
    nogos = add_additional_columns(nogos, ['metric'], ['proportion'])
    
    response_count = process_responses(gonogo_data_pre, 'event_type=="response" or event_type=="response-time-out"', ['trial_type', 'action'], 'value')
    response_count = add_additional_columns(response_count, ['metric'], ['count'])
    
    
    # Combine everything into a long-format dataframe
    long_format_df = pd.concat([
        nogos.assign(participant=participant_id, total_reward=total_reward),
        response_count.assign(participant=participant_id, total_reward=total_reward),
    ], ignore_index=True)

    return long_format_df


def summary_twostep_df(data, participant):
    
    temp_df = data.groupby(["transition_reward"]).stay.mean().reset_index().rename(columns={'transition_reward': 'trial_type', "stay": "value"})
    temp_df = add_additional_columns(temp_df, ["metric"], ["proportion"])

    total_reward = data.query(
        "event_type == 'trial-end'"
    ).total_reward.values[-1]
    resp_order = data.query("event_type == 'trial-end'").response_order.value_counts().reset_index()

    resp_order = resp_order.rename(columns={'response_order': 'trial_type', 'count': 'value'})
    resp_order = add_additional_columns(resp_order, ['metric'], ['count'])
    
    # Combine everything into a long-format dataframe
    long_format_df = pd.concat(
        [
            temp_df.assign(
                participant=participant, total_reward=total_reward
            ),
            resp_order.assign(
                participant=participant, total_reward=total_reward
            ),
        ],
        ignore_index=True,
    )

    return long_format_df