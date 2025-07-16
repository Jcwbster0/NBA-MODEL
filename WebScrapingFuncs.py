import pandas as pd
import time


NBA_SCHED_PATH = ''

MONTH_DICT = {'Jan':'01', 'Feb':'02', 'Mar':'03', 'Apr':'04', 'May':'05', 'Jun':'06', 'Oct':'10', 'Nov':'11', 'Dec':'12'}


NBA_DICT = {'Phoenix Suns':'PHO', 'Memphis Grizzlies':'MEM', 'Golden State Warriors':'GSW', 'Miami Heat':'MIA', 'Dallas Mavericks':'DAL', 'Boston Celtics':'BOS',
            'Milwaukee Bucks':'MIL', 'Philadelphia 76ers':'PHI', 'Utah Jazz':'UTA', 'Denver Nuggets': 'DEN', 'Toronto Raptors':'TOR', 'Chicago Bulls':'CHI', 'Minnesota Timberwolves':'MIN',
            'Brooklyn Nets':'BRK', 'Cleveland Cavaliers':'CLE', 'Atlanta Hawks':'ATL', 'Charlotte Hornets':'CHO', 'Los Angeles Clippers':'LAC', 'New York Knicks':'NYK', 'New Orleans Pelicans':'NOP', 
            'Washington Wizards':'WAS', 'San Antonio Spurs':'SAS', 'Los Angeles Lakers':'LAL', 'Sacramento Kings':'SAC', 'Portland Trail Blazers':'POR', 'Indiana Pacers':'IND', 'Oklahoma City Thunder':'OKC',
            'Detroit Pistons':'DET', 'Orlando Magic':'ORL', 'Houston Rockets':'HOU'}

def get_sched_as_csv(seasons, file_name):
    df = pd.DataFrame()
    for season in seasons:
        for month in ('october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june'):
            season_url = f'https://www.basketball-reference.com/leagues/NBA_{str(season)}_games-{month}.html'
            df = pd.concat([df, pd.read_html(season_url)[0]])
            time.sleep(5)
    df.to_csv(f'{file_name}', index=False)

def csv_to_df(csv):
    return pd.read_csv(csv)

def calc_TS(PTS, FGA, FTA):
    return PTS / (2 * (FGA + 0.44 * FTA))

def calc_AST(AST, FG):
    return AST/FG

def calc_POSS(FGA, TOV, FTA, ORB, DRB, FG, opp_DRB, opp_FGA, opp_FTA, opp_ORB, opp_FG, opp_TOV):
    return 0.5 * ((FGA + 0.4 * FTA - 1.07 * (ORB / (ORB + opp_DRB)) *(FGA - FG) + TOV) + (opp_FGA + 0.4 * opp_FTA - 1.07 * (opp_ORB / (opp_ORB + DRB)) * (opp_FGA - opp_FG) + opp_TOV))

def calc_STL(STL, OPP_POSS):
    return STL/OPP_POSS

def calc_TWPA(FGA, THPA):
    return FGA - THPA

def calc_BLK(BLK, OPP_TWPA):
    return BLK/OPP_TWPA

def calc_ORB(ORB, OPP_DRB):
   return ORB/(ORB+OPP_DRB)

def calc_DRB(DRB, OPP_ORB):
    return DRB/(DRB+OPP_ORB)

def calc_TOV(TOV, FGA, FTA):
    return TOV / (FGA + 0.44 * FTA + TOV)

def calc_ORTG(PTS, POSS):
    return 100 * (PTS/POSS)

def calc_DRTG(OPP_PTS, OPP_POSS):
    return 100 * (OPP_PTS/OPP_POSS)

def calc_GAME_RESULT(home_PTS, away_PTS):
    game_result = 0
    if home_PTS > away_PTS:
        game_result = 0
    else:
        game_result = 1
    return game_result

def convert_date(date):
    month = MONTH_DICT[date[5:8]]
    day = date[9:11]
    if ',' in day:
        day = f'0{day[0]}'
    year = date[-4:]
    return year + month + day + '0'

def build_game_url(row):
    date = convert_date(row[1]['Date'])
    home_team = NBA_DICT[row[1]['Home/Neutral']]
    away_team = NBA_DICT[row[1]['Visitor/Neutral']]
    url = f'https://www.basketball-reference.com/boxscores/{date}{home_team}.html'
    return url, home_team, away_team, date

def get_basic_stats(home_basic, away_basic):
    home_team_row = len(home_basic.index)-1
    away_team_row = len(away_basic.index)-1
    PTS =(int(home_basic.loc[home_team_row,'PTS']),int(away_basic.loc[away_team_row,'PTS']))
    FGA =(int(home_basic.loc[home_team_row,'FGA']),int(away_basic.loc[away_team_row,'FGA']))
    FTA =(int(home_basic.loc[home_team_row,'FTA']),int(away_basic.loc[away_team_row,'FTA']))
    AST =(int(home_basic.loc[home_team_row,'AST']),int(away_basic.loc[away_team_row,'AST']))
    FG =(int(home_basic.loc[home_team_row,'FG']),int(away_basic.loc[away_team_row,'FG']))
    TOV =(int(home_basic.loc[home_team_row,'TOV']),int(away_basic.loc[away_team_row,'TOV']))
    ORB_raw =(int(home_basic.loc[home_team_row,'ORB']),int(away_basic.loc[away_team_row,'ORB']))
    DRB_raw =(int(home_basic.loc[home_team_row,'DRB']), int(away_basic.loc[away_team_row,'DRB']))
    POSS =(calc_POSS(int(FGA[0]),int(TOV[0]),int(FTA[0]),int(ORB_raw[0]),int(DRB_raw[0]),int(FG[0]),int(DRB_raw[1]),int(FGA[1]),int(FTA[1]),int(ORB_raw[1]),int(FG[1]),int(TOV[1])), 
           calc_POSS(int(FGA[1]),int(TOV[1]),int(FTA[1]),int(ORB_raw[1]),int(DRB_raw[1]),int(FG[1]),int(DRB_raw[0]),int(FGA[0]),int(FTA[0]),int(ORB_raw[0]),int(FG[0]),int(TOV[0])))
    STL =(int(home_basic.loc[home_team_row,'STL']),int(away_basic.loc[away_team_row,'STL']))
    BLK =(int(home_basic.loc[home_team_row,'BLK']),int(away_basic.loc[away_team_row,'BLK']))
    THPA =(int(home_basic.loc[home_team_row,'3PA']),int(away_basic.loc[away_team_row,'3PA']))
    return PTS, FGA, FTA, AST, FG, TOV, ORB_raw, DRB_raw, POSS, STL, BLK, THPA

def calc_adv_stats(basic_stats):
    PTS, FGA, FTA, AST, FG, TOV, ORB_raw, DRB_raw, POSS, STL, BLK, THPA = basic_stats
    home_TS = calc_TS(PTS[0], FGA[0], FTA[0])
    home_AST = calc_AST(AST[0], FG[0])
    home_STL = calc_STL(STL[0], POSS[1])
    home_TWPA = calc_TWPA(FGA[0], THPA[0])
    home_BLK = calc_BLK(BLK[0], calc_TWPA(FGA[1],THPA[1]))
    home_ORB = calc_ORB(ORB_raw[0],DRB_raw[1])
    home_DRB = calc_DRB(DRB_raw[0],ORB_raw[1])
    home_TOV = calc_TOV(TOV[0], FGA[0], FTA[0])
    home_ORTG = calc_ORTG(PTS[0], POSS[0])
    home_DRTG = calc_DRTG(PTS[1], POSS[1])
    GAME_RESULT = calc_GAME_RESULT(PTS[0], PTS[1])

    away_TS = calc_TS(PTS[1], FGA[1], FTA[1])
    away_AST = calc_AST(AST[1], FG[1])
    away_STL = calc_STL(STL[1], POSS[0])
    away_TWPA = calc_TWPA(FGA[1], THPA[1])
    away_BLK = calc_BLK(BLK[1], calc_TWPA(FGA[0],THPA[0]))
    away_ORB = calc_ORB(ORB_raw[1],DRB_raw[0])
    away_DRB = calc_DRB(DRB_raw[1],ORB_raw[0])
    away_TOV = calc_TOV(TOV[1], FGA[1], FTA[1])
    away_ORTG = calc_ORTG(PTS[1], POSS[1])
    away_DRTG = calc_DRTG(PTS[0], POSS[0])
    stats_dict = {'home_TS%':home_TS,'home_AST%':home_AST,'home_STL%':home_STL,'home_BLK%':home_BLK,'home_ORB%':home_ORB,'home_DRB%':home_DRB,'home_TOV%':home_TOV,'home_ORTG':home_ORTG,'home_DRTG':home_DRTG,
                  'home_PTS':PTS[0], 'home_FGA':FGA[0], 'home_FTA':FTA[0], 'home_AST':AST[0], 'home_FG':FG[0],'home_TOV':TOV[0], 'home_ORB': ORB_raw[0], 'home_DRB':DRB_raw[0],'home_POSS':POSS[0],'home_STL':STL[0],'home_BLK':BLK[0],'home_THPA':THPA[0],
                  'away_TS%':away_TS,'away_AST%':away_AST,'away_STL%':away_STL,'away_BLK%':away_BLK,'away_ORB%':away_ORB,'away_DRB%':away_DRB,'away_TOV%':away_TOV,'away_ORTG':away_ORTG,'away_DRTG':away_DRTG, 'GAME_RESULT':GAME_RESULT,
                  'away_PTS':PTS[1], 'away_FGA':FGA[1], 'away_FTA':FTA[1], 'away_AST':AST[1], 'away_FG':FG[1],'away_TOV':TOV[1], 'away_ORB': ORB_raw[1], 'away_DRB':DRB_raw[1],'away_POSS':POSS[1],'away_STL':STL[1],'away_BLK':BLK[1],'away_THPA':THPA[1]}
    return stats_dict
    
def create_game_df(csv_path, file_name):
    df = csv_to_df(csv_path)
    training_df = pd.DataFrame()
    counter = 0
    for row in df.iterrows():
        game_url, home_team, away_team, date = build_game_url(row)
        try:
            game_dfs = pd.read_html(game_url)
            time.sleep(5)
            home_idx = 8 + int((len(game_dfs) - 16) / 2)
            away_basic = game_dfs[0].droplevel(0, axis=1)
            home_basic = game_dfs[home_idx].droplevel(0,axis=1)
            adv_stats = calc_adv_stats(get_basic_stats(home_basic, away_basic))
            adv_stats['home_team'] = home_team
            adv_stats['away_team'] = away_team
            adv_stats['url'] = game_url
            adv_stats['Date'] = date
            curr_game_df = pd.DataFrame(adv_stats, index=[0])
            training_df = pd.concat([training_df, curr_game_df])
            counter += 1
            print(counter)
        except Exception as e:
            print(f'ERROR FOR GAME NUMBER {counter}: {e}')
            print(game_url)
            time.sleep(5)
    training_df.to_csv(f'{file_name}', index=False)
    print('success')