"""Optional, read-only market comparison using The Odds API v4.

An API key is supplied at runtime only (never written to disk). These prices
are for research. No wagering endpoints, no assertions of betting value.
"""
from __future__ import annotations
from datetime import datetime, timezone
import requests
import pandas as pd
from data_sources import normalize_name

URL = 'https://api.the-odds-api.com/v4/sports/basketball_nba'
MARKETS = {
    'player_points': 'PTS', 'player_rebounds': 'REB',
    'player_assists': 'AST', 'player_threes': '3PM',
    'player_blocks': 'BLK', 'player_steals': 'STL',
    'player_points_rebounds_assists': 'PRA',
}


def _request(path: str, api_key: str, params: dict) -> list | dict:
    if not api_key or len(api_key.strip()) < 8:
        raise ValueError('A valid Odds API key is required')
    resp = requests.get(URL + path, params={**params, 'apiKey': api_key.strip()}, timeout=18)
    resp.raise_for_status()
    return resp.json()


def fetch_events(api_key: str) -> list[dict]:
    data = _request('/events', api_key, {})
    if not isinstance(data, list):
        raise ValueError('Unexpected event response from odds provider')
    return data


def fetch_event_odds(event_id: str, api_key: str, props: bool = True, region: str = 'us') -> dict:
    market_keys = ','.join(MARKETS) if props else 'h2h,spreads,totals'
    data = _request(f'/events/{event_id}/odds', api_key,
                    {'regions': region, 'markets': market_keys, 'oddsFormat': 'american'})
    if not isinstance(data, dict):
        raise ValueError('Unexpected event odds format')
    return data


def parse_props(payload: dict) -> pd.DataFrame:
    rows = []
    for book in payload.get('bookmakers', []):
        for market in book.get('markets', []):
            if market.get('key') not in MARKETS:
                continue
            for outcome in market.get('outcomes', []):
                name = str(outcome.get('name', ''))
                if name not in ('Over', 'Under'):
                    continue
                if outcome.get('point') is None or outcome.get('price') is None:
                    continue
                rows.append({'Book': book.get('title', book.get('key', '')),
                    'Player': outcome.get('description', ''),
                    'Stat': MARKETS[market['key']],
                    'Line': float(outcome['point']),
                    'Side': name, 'American odds': int(outcome['price']),
                    'Updated': market.get('last_update', book.get('last_update', ''))})
    return pd.DataFrame(rows)


def american_probability(price: int) -> float:
    """Bookmaker-implied chance before margin. Moneyline value not a win guarantee."""
    if price == 0:
        raise ValueError('Zero price is invalid')
    return 100/(price+100) if price > 0 else -price/(-price+100)


def return_per_dollar(price: int) -> float:
    return price/100 if price > 0 else 100 / abs(price)


def compare_props(result: dict, quotes: pd.DataFrame) -> pd.DataFrame:
    """Compare probabilities at supplied line, avoid claiming a calibrated edge."""
    if quotes.empty:
        return pd.DataFrame()
    keys = {normalize_name(k.split('|', 1)[1]): k for k in result['samples']}
    output = []
    # Normalize an over/under pair from the same bookmaker, player, market, and line.
    # Without BOTH sides, no credible margin-stripped probability is available.
    fair = {}
    for (book, athlete, stat, line), pair in quotes.groupby(['Book', 'Player', 'Stat', 'Line']):
        sides = {str(q['Side']): int(q['American odds']) for _, q in pair.iterrows()}
        if 'Over' in sides and 'Under' in sides:
            over, under = american_probability(sides['Over']), american_probability(sides['Under'])
            if over + under > 0:
                fair[(book, athlete, stat, line)] = {'Over': over/(over+under), 'Under':under/(over+under)}
    for _, q in quotes.iterrows():
        player_id = keys.get(normalize_name(q['Player']))
        if player_id is None or q['Stat'] not in result['samples'][player_id]:
            continue
        samples = result['samples'][player_id][q['Stat']]
        is_over = q['Side'] == 'Over'
        p = float((samples > q['Line']).mean() if is_over else (samples < q['Line']).mean())
        push = float((samples == q['Line']).mean())
        implied = american_probability(int(q['American odds']))
        fair_price = fair.get((q['Book'], q['Player'], q['Stat'], q['Line']), {}).get(q['Side'])
        # Push fraction has zero profit. Same-line settlement handled separately.
        ev = p * return_per_dollar(int(q['American odds'])) - (1-p-push)
        output.append({**q.to_dict(), 'Model probability': round(p, 4),
                       'Raw implied probability': round(implied, 4),
                       'No-vig fair probability': round(float(fair_price), 4) if fair_price is not None else None,
                       'Fair probability gap': round(p - fair_price, 4) if fair_price is not None else None,
                       'Raw probability gap': round(p - implied, 4),
                       'Hypothetical $1 EV': round(ev, 4), 'Push chance': round(push, 4),
                       'Matched model player': player_id.split('|', 1)[1]})
    return pd.DataFrame(output).sort_values('Hypothetical $1 EV', ascending=False) if output else pd.DataFrame()


def match_event(events: list[dict], away_name: str, home_name: str) -> list[dict]:
    return [e for e in events if normalize_name(e.get('away_team', '')) == normalize_name(away_name)
            and normalize_name(e.get('home_team', '')) == normalize_name(home_name)]
