# %%
import dhc_maintenance.toolkit as toolkit
import dhc_maintenance.simulation as simulation
import dhc_maintenance.maintenance as maintenance
import dhc_maintenance.pipe as pipe
# %%
pipe1 = pipe.KMR(
    ID=123456,
    TYPE=pipe.PipeSystem["KMR"],
    medium_count=pipe.MedPipeCount(1),
    dn=150,
    laying=pipe.LayingSystem["Burried"],
    length=10,
    flow=pipe.Flow["Supply"],
    build_year=1988,
    connection=pipe.PipeConnection["Line"],
    life_status=pipe.Status["InOperation"],
    failure_years=[0],
    failure_degrees=[pipe.FailureLevels["NoFailure"]],
    failure_types=[pipe.FailureType["NoFailure"]],
    decommission_year=0,
)
# %%
pipe1.cycle_count()