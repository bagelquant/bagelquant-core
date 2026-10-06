from datetime import date, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bagelquant_core import Domain, Node, capture_operator_checkpoints
from bagelquant_core.operator.rolling_stats import ewm_mean, ewm_std, ewm_var


@pytest.mark.parametrize('operator',[ewm_mean,ewm_var,ewm_std])
@pytest.mark.parametrize('adjust',[False,True])
@pytest.mark.parametrize('ignore_na',[False,True])
@pytest.mark.parametrize('bias',[False,True])
def test_exponential_checkpoint_matches_polars_and_full_prefix(operator,adjust,ignore_na,bias):
    days=[date(2024,1,2)+timedelta(days=index) for index in range(12)]
    values=[None,2.0,3.0,None,4.0,5.0,None,None,6.0,7.0,8.0,None]
    frame=pl.DataFrame({'time':days*2,'asset_id':['A']*12+['B']*12,'value':values+values[::-1]})
    parameters={'alpha':0.4,'adjust':adjust,'ignore_na':ignore_na,'min_periods':2}
    if operator!=ewm_mean:
        parameters['bias']=bias
    def graph(selected):
        domain=Domain(calendar=selected['time'].unique().sort(),universe=['A','B'])
        return operator(Node.from_domain(selected,domain,source_key='observations'),name='exp',**parameters)
    reference=graph(frame).compute().collect().sort(['time','asset_id'])
    with capture_operator_checkpoints() as first:
        prefix=graph(frame.filter(pl.col('time')<=days[6])).compute().collect()
    with capture_operator_checkpoints(first.captured):
        suffix=graph(frame.filter(pl.col('time')>=days[4])).compute().collect().filter(pl.col('time')>days[6])
    actual=pl.concat([prefix,suffix]).sort(['time','asset_id'])
    assert_frame_equal(actual,reference,check_exact=False,rel_tol=1e-12,abs_tol=1e-12)
