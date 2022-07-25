import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

match = pd.read_csv('matches.csv')
delivery = pd.read_csv('deliveries.csv')

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

st.sidebar.image('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRw-lfTHk_tbkl0_O6KYhLw1E34tKJcRIeWSca6MQzPxl_Oj8c4IJ-4cm1yNTJr5FOBgEo&usqp=CAU')
user_menu = st.sidebar.radio(
    'MENU:',
    ('Predict score ', 'Overall Analysis', 'Team wise Analysis', 'Player wise Analysis'))

LogReg = pickle.load(open('LogReg.pkl', 'rb'))
Rf = pickle.load(open('Rf.pkl', 'rb'))
dt_clf = pickle.load(open('dt_clf.pkl', 'rb'))


# st.title('Predict The Winning IPL Team')


# Load data
@st.cache(allow_output_mutation=True)
def load_data(nrows):
    data = pd.read_csv('matches.csv', nrows=nrows)
    return data


match = load_data(756)

match['team1'] = match['team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match['team2'] = match['team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
match['winner'] = match['winner'].str.replace('Delhi Daredevils', 'Delhi Capitals')

match['team1'] = match['team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match['team2'] = match['team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
match['winner'] = match['winner'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

# st.image('static/images/IPL_background_image.jpg')

# st.markdown(
#     """
#     <style>
#     .stApp {
#         background: url("https://wallpapercave.com/wp/wp4059913.jpg");
#         background-size: cover;

#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )
if user_menu == 'Predict score ':
    st.title('Predict The Winning IPL Team')
    # st.image('static/images/IPL_background_image.jpg')
    st.image('https://etimg.etb2bimg.com/photo/74508790.cms')
    classifier_name = st.sidebar.selectbox(
        'Select classifier',
        ('Logistic Regression', 'Random Forest')
    )

    col1, col2 = st.columns(2)

    with col1:
        batting_team = st.selectbox('Select the batting Team', sorted(teams))

    with col2:
        bowling_team = st.selectbox('Select the bowling Team', sorted(teams))

    selected_city = st.selectbox('Select host city', sorted(cities))
    target = st.number_input('Target')

    col3, col4, col5 = st.columns(3)

    with col3:
        score = st.number_input('Score')

    with col4:
        overs = st.number_input('Overs Completed')

    with col5:
        wickets_out = st.number_input('Wicket out')

    if st.button('Predict Probability'):
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets_left = 10 - wickets_out
        crr = score / overs
        rrr = (runs_left * 6) / (balls_left)

        input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team],
                                 'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left],
                                 'wickets_left': [wickets_left], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

        # st.table(input_df)
        if classifier_name == "Logistic Regression":
            result = LogReg.predict_proba(input_df)
        elif classifier_name == "Random Forest":
            result = Rf.predict_proba(input_df)
        else:
            result = dt_clf.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        # st.header(batting_team + "-" + str(round(win*100)) +'%')
        # st.header(bowling_team + "-" + str(round(loss*100)) +'%')

        labels = [batting_team, bowling_team]
        sizes = [win, loss]
        explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig1, ax1 = plt.subplots(figsize=(2, 2))
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=True, startangle=90)
        ax1.set_title('Winning Probability of Each Team')
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        st.pyplot(fig1)

# ============================================================================================================================
#  **************************************        OVERALL ANALYSIS        ************************************
# ============================================================================================================================

if user_menu == 'Overall Analysis':
    # st.image('static/images/overall_analysis.jpeg')
    st.title('Overall Analysis')
    ques = st.radio(
        "Select:",
        ('Graphical Visualiztion', 'Tabular View'))
    match_per_season = match.groupby(['Season'])['id'].count().reset_index().rename(columns={'id': 'matches'})

# ============================================================================================================================
    if ques == 'Tabular View':
        # ===================================================================================
        # ************************      MATCHES PER SEASON TABLE   *************************
        # ===================================================================================

        st.title('Number of Matches played per Season')
        st.write(match_per_season)

        # ===================================================================================
        # *************************       TOSS WINNERS TABLE      **************************
        # ===================================================================================
        st.title('Toss Winner(%)')
        current_teams = ['Mumbai Indians', 'Kolkata Knight Riders', 'Chennai Super Kings', 'Kings XI Punjab',
                         'Royal Challengers Bangalore', 'Rajasthan Royals', 'Delhi Capitals', 'Sunrisers Hyderabad']

        # Creating a new dataframe
        toss_winners = pd.DataFrame()

        # Get the number of tosses won
        toss_winners['Toss Won'] = match['toss_winner'].value_counts()
        toss_winners.index.names = ['Teams']
        toss_winners.reset_index(inplace=True)

        toss_winners = toss_winners[toss_winners['Teams'].isin(current_teams)]
        st.write(toss_winners)

        # ===================================================================================
        # *************************     TEAM TOSS_WINNER TABLE    **************************
        # ===================================================================================
        st.title('Total Toss won by Teams')
        toss_winner = match.groupby(['team1'])['toss_winner'].count().reset_index().rename(
            columns={'team1': 'Team', 'toss_winner': 'toss_winner'})
        toss_winner = toss_winner.sort_values(by=['toss_winner'], ascending=False)
        st.write(toss_winner)

    # ============================================================================================================================
    if ques == 'Graphical Visualiztion':

        # ===================================================================================
        # *************************     Season wise performance   **************************
        # ===================================================================================
        st.title('Season wise performance')
        col1, col2 = st.columns(2)
        with col1:

            season = st.selectbox(
                'Select Season',
                ('IPL-2008', 'IPL-2009', 'IPL-2010', 'IPL-2011', 'IPL-2012', 'IPL-2013', 'IPL-2014', 'IPL-2015',
                 'IPL-2016',
                 'IPL-2017', 'IPL-2018', 'IPL-2019'))

        if st.button('Performance of Teams in selected Season'):
            match_wins = match[match['Season'] == season]['winner'].value_counts()

            plt.figure(figsize=(10, 7))
            ax = sns.barplot(match_wins.index, match_wins.values, alpha=0.8)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
            plt.title('Performance of Each Team')
            plt.ylabel('Number of Match Wins in the Season', fontsize=12)
            plt.xlabel('Teams', fontsize=12)
            plt.tight_layout()
            plt.show()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

        # ===================================================================================
        # *************************    MATCHES PER SEASON GRAPH   **************************
        # ===================================================================================
        st.title('Number of Matches played per Season')
        plt.figure(figsize=(10, 7))
        ax = sns.barplot(match_per_season.Season, match_per_season.matches, alpha=0.8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        plt.title('Number of matches played in different seasons')
        plt.ylabel('Matches', fontsize=12)
        plt.xlabel('Seasons', fontsize=12)
        plt.tight_layout()
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # ===================================================================================
        # *************************    Toss Winner(%) GRAPH   **************************
        # ===================================================================================
        st.title('Toss Winner(%)')
        current_teams = ['Mumbai Indians', 'Kolkata Knight Riders', 'Chennai Super Kings', 'Kings XI Punjab',
                         'Royal Challengers Bangalore', 'Rajasthan Royals', 'Delhi Capitals', 'Sunrisers Hyderabad']

        toss_winners = pd.DataFrame()

        # Get the number of tosses won
        toss_winners['Toss Won'] = match['toss_winner'].value_counts()
        toss_winners.index.names = ['Teams']
        toss_winners.reset_index(inplace=True)

        toss_winners = toss_winners[toss_winners['Teams'].isin(current_teams)]

        plt.figure(figsize=(16, 10))
        font = {'color': 'darkcyan',
                'weight': 'bold',
                'size': 30,
                }
        colors = ['#15244C', '#FFFF48', '#292734', '#EF2920', '#CD202D', '#ECC5F2', '#294A73', '#D4480B', '#242307',
                  '#FD511F', '#158EA6', '#E82865', '#005DB7', '#C23E25', '#E82865']
        textprops = {"fontsize": 15, "color": "teal"}
        plt.title('Toss Winners', fontdict=font)
        plt.pie(toss_winners['Toss Won'], labels=toss_winners['Teams'], autopct='%1.1f%%', startangle=140,
                textprops=textprops, colors=colors)

        plt.axis('equal')
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # col1 ,col2 = st.columns([0.3,0.9])
        # with col1:
        #     st.write("kjrnefkewfjcoewncdfj\nhjbdrfcvri\njfhcbew\nrfjvbwofjwed")
        # with col2:
        #     # Plot

        # ===================================================================================
        # *************************    TOSS DECISION GRAPH   **************************
        # ===================================================================================

        st.title('Toss Decision Statistics')
        fig = plt.gcf()
        fig.set_size_inches(18.5, 8)
        sns.countplot(match['toss_winner'], order=match['toss_winner'].value_counts().index, palette='Set2',
                      hue=match['toss_decision'])
        plt.title('Toss decision statistics for all the team', fontsize=15)
        plt.yticks(fontsize=15)
        plt.xticks(fontsize=15, rotation=90)
        plt.xlabel('Toss winner', fontsize=15)
        plt.ylabel('Count', fontsize=15)
        plt.legend(['Field first', 'Bat first'], loc='best', fontsize=15)
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # ===================================================================================
        # *************************    TEAM TOSS_WINNER GRAPH   **************************
        # ===================================================================================
        st.title('Total Toss won by Teams')
        toss_winner = match.groupby(['team1'])['toss_winner'].count().reset_index().rename(
            columns={'team1': 'Team', 'toss_winner': 'toss_winner'})
        toss_winner = toss_winner.sort_values(by=['toss_winner'], ascending=False)

        plt.figure(figsize=(10, 7))
        ax = sns.barplot(toss_winner.Team, toss_winner.toss_winner, alpha=0.8, color='blue')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
        plt.title('Number of tosses won by teams')
        plt.ylabel('Count', fontsize=12)
        plt.xlabel('Teams', fontsize=12)
        plt.tight_layout()
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

# ============================================================================================================================
#  **************************************        TEAM WISE ANALYSIS        ************************************
# ============================================================================================================================


if user_menu == 'Team wise Analysis':
    st.title('Team Analysis')
    ques = st.radio(
        "Select:",
        ('Graphical Visualiztion', 'Tabular View'))

    # ============================================================================================================================
    if ques == 'Tabular View':
        # ===================================================================================
        # *****************    Total no. of matches won(2008-2019)   ***********************
        # ===================================================================================

        st.title('Total no. of matches won(2008-2019)')
        wins = pd.DataFrame(match['winner'].value_counts())
        st.write(wins)

    # ============================================================================================================================
    if ques == 'Graphical Visualiztion':

        # ===================================================================================
        # **********    Total no. of matches won(2008-2019)  GRAPH  ***********************
        # ===================================================================================

        st.title('Total no. of matches won(2008-2019)')
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        wins = pd.DataFrame(match['winner'].value_counts())
        wins['name'] = wins.index
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=16)
        plt.bar(wins['name'],
                wins['winner'],
                color=['#15244C', '#FFFF48', '#292734', '#EF2920', '#CD202D', '#ECC5F2',
                       '#294A73', '#D4480B', '#242307', '#FD511F', '#158EA6', '#E82865',
                       '#005DB7', '#C23E25', '#E82865']
                , alpha=0.8)
        count = 0
        for i in wins['winner']:
            plt.text(count - 0.15, i - 4, str(i), size=15, color='lightgreen', rotation=90)
            count += 1
        plt.title('Total wins by each team', fontsize=20)
        plt.xlabel('Teams', fontsize=15)
        plt.ylabel('Total no. of matches won(2008-2019)', fontsize=14)
        plt.show()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

    # ===================================================================================
    # *****************    Total no. of matches won(2008-2019)   ***********************
    # ===================================================================================
    st.title('Total runs in First Innings')
    runs = delivery.groupby(['match_id', 'inning', 'batting_team'])[['total_runs']].sum().reset_index()
    runs.drop('match_id', axis=1, inplace=True)
    runs.head()
    inning1 = runs[runs['inning'] == 1]
    inning2 = runs[runs['inning'] == 2]

    # BATTING FIRST-1 INNING
    sns.boxplot(y='total_runs', x='batting_team', data=inning1)
    plt.xticks(rotation=90)
    # plt.yticks(ticks, ticks)
    plt.xticks(fontsize=15)
    plt.title("1 Innings'(Batting first)'", fontsize=20)
    plt.ylabel('Total_runs', fontsize=20)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # Bowling FIRST-2 INNING
    st.title('Total runs in Second Innings')
    sns.boxplot(y='total_runs', x='batting_team', data=inning2)
    plt.xticks(rotation=90)
    # plt.yticks(ticks, ticks)
    plt.xticks(fontsize=15)
    plt.title("2 Innings'(Bowling first)'", fontsize=20)
    plt.ylabel('Total_runs', fontsize=20)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # ===================================================================================
    # *****************    STADIUM WISE ANALYSIS PER TEAM GRAPH   ***********************
    # ===================================================================================
    team_name = st.selectbox(
        'Select Team Name',
        teams)


    def lucky(match_data, team_name):
        return match_data[match_data['winner'] == team_name]['venue'].value_counts().nlargest(10)

    mi = lucky(match, team_name)
    values = mi
    labels = mi.index

    # ----------------    Plot   ------------------------
    plt.figure(figsize=(40, 35))
    font = {'color': 'darkcyan',
            'weight': 'bold',
            'size': 38,
            }
    colors = ['#FF6701', '#E8F9FD', '#99C4C8', '#BABD42', '#DEB6AB', '#B4E197', '#E9D5DA', '#FBD6D2', '#39AEA9',
              '#FFE61B', '#74959A', '#A2B38B', '#C1DEAE', '#B980F0', '#FF6701']
    textprops = {"fontsize": 35, "color": "black"}
    plt.title('Stadium and Max. Win at Stadiums', fontdict=font)
    plt.pie(x=values, labels=labels, autopct='%1.1f%%', startangle=60, textprops=textprops, colors=colors)

    plt.axis('equal')
    plt.show()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

# ============================================================================================================================
#  **************************************        PLAYER WISE ANALYSIS        ************************************
# ============================================================================================================================
if user_menu == 'Player wise Analysis':
    st.title('Player wise Analysis')
    ques = st.radio(
        "Select:",
        ('Graphical Visualiztion', 'Tabular View'))
    match_per_season = match.groupby(['Season'])['id'].count().reset_index().rename(columns={'id': 'matches'})
    # match_per_season.style.background_gradient(cmap='GnBu')

# ============================================================================================================================
    if ques == 'Tabular View':

        # ===================================================================================
        # *****************            TOP 10 BatsMan                 ***********************
        # ===================================================================================

        st.title('Top 10 Batsman')
        runs = delivery.groupby(['batsman'])['batsman_runs'].sum().reset_index()
        runs.columns = ['Batsman', 'runs']
        top_batsman = runs.sort_values(by='runs', ascending=False).head(10).reset_index().drop('index', axis=1)
        # top_batsman .style.background_gradient(cmap='PuBu')
        st.write(top_batsman)

        # ===================================================================================
        # *****************            TOP 6s by BatsMan              ***********************
        # ===================================================================================
        st.title('Most 6s by Player')
        six = delivery.groupby('batsman')['batsman_runs'].agg(lambda x: (x == 6).sum()).reset_index()
        six.columns = ['Batsman', '6s']
        player_6 = six.sort_values(by='6s', ascending=False).head(10).reset_index().drop('index', axis=1)
        # player_6.style.background_gradient(cmap='PuBu')
        st.write(player_6)

        # ===================================================================================
        # *****************            TOP 4s by BatsMan              ***********************
        # ===================================================================================
        st.title('Most 4s by Player')
        fours = delivery.groupby('batsman')['batsman_runs'].agg(lambda x: (x == 4).sum()).reset_index()
        fours.columns = ['Batsman', '4s']
        player_4 = fours.sort_values(by='4s', ascending=False).head(10).reset_index().drop('index', axis=1)
        # z.style.background_gradient(cmap='PuBu')
        st.write(player_4)

        # ===================================================================================
        # *****************            TOP Man of the Match           ***********************
        # ===================================================================================
        st.title('Man of the Match')
        Player_of_match = pd.DataFrame(
            match.groupby('player_of_match').count()['id'].sort_values(ascending=False).head(10),
            columns=['id']).reset_index()
        st.write(Player_of_match)

# ============================================================================================================================
    if ques == 'Graphical Visualiztion':

        # ===================================================================================
        # *****************            TOP 10 BatsMan                 ***********************
        # ===================================================================================
        st.title('Top 10 Batsman')
        runs = delivery.groupby(['batsman'])['batsman_runs'].sum().reset_index()
        runs.columns = ['Batsman', 'runs']
        top_batsman = runs.sort_values(by='runs', ascending=False).head(10).reset_index().drop('index', axis=1)

        # ----------------    Plot   ------------------------
        plt.figure(figsize=(15, 10))
        ticks = [1000, 2000, 3000, 4000, 5000]
        # base_color = sns.color_palette("flare", as_cmap=True)
        sns.barplot(y='runs', x='Batsman', data=top_batsman, color='#006E7F')
        plt.xticks(rotation=90)
        plt.yticks(ticks, ticks)
        plt.xticks(fontsize=15)
        plt.title('TOP 10 BATSMAN', fontsize=20)
        plt.ylabel('Runs', fontsize=20)
        plt.xlabel('Players', fontsize=20)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # ===================================================================================
        # *****************            TOP 6s by BatsMan              ***********************
        # ===================================================================================
        st.title('Most 6s by Player')
        six = delivery.groupby('batsman')['batsman_runs'].agg(lambda x: (x == 6).sum()).reset_index()
        six.columns = ['Batsman', '6s']
        player_6 = six.sort_values(by='6s', ascending=False).head(10).reset_index().drop('index', axis=1)

        # ----------------    Plot   ------------------------
        plt.figure(figsize=(15, 10))
        ticks = [100, 150, 200, 250, 300]
        # base_color = sns.color_palette()[3]
        sns.barplot(y='6s', x='Batsman', data=player_6, color='#F55353')
        plt.xticks(rotation=90)
        plt.yticks(ticks, ticks)
        plt.xticks(fontsize=15)
        plt.title("TOP 6's by BATSMAN", fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.xlabel('Players', fontsize=20)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # ===================================================================================
        # *****************            TOP 4s by BatsMan              ***********************
        # ===================================================================================
        st.title('Most 4s by Player')
        fours = delivery.groupby('batsman')['batsman_runs'].agg(lambda x: (x == 4).sum()).reset_index()
        fours.columns = ['Batsman', '4s']
        player_4 = fours.sort_values(by='4s', ascending=False).head(10).reset_index().drop('index', axis=1)

        # ----------------    Plot   ------------------------
        plt.figure(figsize=(15, 10))
        ticks = [100, 150, 200, 250, 300, 350, 400, 450, 500]
        # base_color = sns.color_palette()[3]
        sns.barplot(y='4s', x='Batsman', data=player_4, color='#EE5007')
        plt.xticks(rotation=90)
        plt.yticks(ticks, ticks)
        plt.xticks(fontsize=15)
        plt.title("TOP 4's by BATSMAN", fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.xlabel('Players', fontsize=20)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # ===================================================================================
        # *****************           Leading wicket-takers         ***********************
        # ===================================================================================
        st.title('Leading Wicket Takers')
        delivery['dismissal_kind'].unique()
        dismissal_kinds = ['caught', 'bowled', 'lbw', 'caught and bowled',
                           'stumped', 'hit wicket']
        hwt = delivery[delivery["dismissal_kind"].isin(dismissal_kinds)]
        bo = hwt['bowler'].value_counts()

        # ----------------    Plot   ------------------------
        plt.figure(figsize=(15, 10))
        ticks = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        # base_color = sns.color_palette()[3]
        sns.barplot(x=bo[:10].index, y=bo[:10], data=player_4, color='#36AE7C')
        plt.xticks(rotation=90)
        plt.yticks(ticks, ticks)
        plt.xticks(fontsize=15)
        plt.title("Leading wicket-takers", fontsize=20)
        plt.ylabel('Total Wickets', fontsize=20)
        plt.xlabel('Bowlers', fontsize=20)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # ===================================================================================
        # *****************            TOP Man of the Match           ***********************
        # ===================================================================================
        st.title('Man of the Match')
        Player_of_match = pd.DataFrame(
            match.groupby('player_of_match').count()['id'].sort_values(ascending=False).head(10),
            columns=['id']).reset_index()

        # ----------------    Plot   ------------------------
        plt.figure(figsize=(15, 10))
        ticks = [0, 3, 6, 9, 12, 15, 18, 21]
        # base_color = sns.color_palette()[3]
        sns.barplot(y='id', x='player_of_match', data=Player_of_match, color='#333C83')
        plt.xticks(rotation=90)
        plt.yticks(ticks, ticks)
        plt.xticks(fontsize=15)
        plt.title('Won Player of Match title', fontsize=20)
        plt.ylabel('Count', fontsize=20)
        plt.xlabel('Players', fontsize=20)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()

        # ===================================================================================
        # *****************            3 PLAYER PROFILE            ***********************
        # ===================================================================================
        st.title('Player wise Analysis')
        ques = st.radio(
            "Select any Player:",
            ('Virat Kholi', 'M S Dhoni', 'AB de Villiers'))

        # ----------------    PLAYER 1  ------------------------
        if ques == 'Virat Kholi':
            filt = (delivery['batsman'] == 'V Kohli')
            df_kohli = delivery[filt]
            df_kohli.head()
            len(df_kohli[df_kohli['batsman_runs'] == 4])
            len(df_kohli[df_kohli['batsman_runs'] == 6])
            df_kohli['total_runs'].sum()

            def count(df_kohli, runs):
                return len(df_kohli[df_kohli['batsman_runs'] == runs]) * runs

            st.write("Player Profile -V Kohli")
            st.write('Runs Scored:')
            col1, col2, col3 = st.columns(3)

            with col1:
                st.header('By 1')
                st.title(count(df_kohli, 1))
            with col2:
                st.header('By 2')
                st.title(count(df_kohli, 2))
            with col3:
                st.header('By 3')
                st.title(count(df_kohli, 3))

            col4, col5, col6 = st.columns(3)
            with col4:
                st.header('By 4')
                st.title(count(df_kohli, 4))
            with col5:
                st.header('By 6')
                st.title(count(df_kohli, 6))
            with col6:
                st.header('Man of Match')
                st.title(12)

        # ----------------    PLAYER 2  ------------------------
        if ques == 'M S Dhoni':
            filt = (delivery['batsman'] == 'MS Dhoni')
            df_kohli = delivery[filt]
            df_kohli.head()
            len(df_kohli[df_kohli['batsman_runs'] == 4])
            len(df_kohli[df_kohli['batsman_runs'] == 6])
            df_kohli['total_runs'].sum()

            def count(df_kohli, runs):
                return len(df_kohli[df_kohli['batsman_runs'] == runs]) * runs

            st.write("Player Profile -MS Dhoni")
            st.write('Runs Scored:')
            col1, col2, col3 = st.columns(3)

            with col1:
                st.header('By 1')
                st.title(count(df_kohli, 1))
            with col2:
                st.header('By 2')
                st.title(count(df_kohli, 2))
            with col3:
                st.header('By 3')
                st.title(count(df_kohli, 3))

            col4, col5, col6 = st.columns(3)
            with col4:
                st.header('By 4')
                st.title(count(df_kohli, 4))
            with col5:
                st.header('By 6')
                st.title(count(df_kohli, 6))
            with col6:
                st.header('Man of Match')
                st.title(17)

        # ----------------    PLAYER 3  ------------------------
        if ques == 'AB de Villiers':
            filt = (delivery['batsman'] == 'AB de Villiers')
            df_kohli = delivery[filt]
            df_kohli.head()
            len(df_kohli[df_kohli['batsman_runs'] == 4])
            len(df_kohli[df_kohli['batsman_runs'] == 6])
            df_kohli['total_runs'].sum()

            def count(df_kohli, runs):
                return len(df_kohli[df_kohli['batsman_runs'] == runs]) * runs

            st.write("Player Profile -AB de Villiers")
            st.write('Runs Scored:')
            col1, col2, col3 = st.columns(3)

            with col1:
                st.header('By 1')
                st.title(count(df_kohli, 1))
            with col2:
                st.header('By 2')
                st.title(count(df_kohli, 2))
            with col3:
                st.header('By 3')
                st.title(count(df_kohli, 3))

            col4, col5, col6 = st.columns(3)
            with col4:
                st.header('By 4')
                st.title(count(df_kohli, 4))
            with col5:
                st.header('By 6')
                st.title(count(df_kohli, 6))
            with col6:
                st.header('Man of Match')
                st.title(20)
