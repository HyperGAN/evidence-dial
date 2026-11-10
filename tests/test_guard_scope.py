import json
from pathlib import Path
import pytest
from dataclasses import asdict
from particle_sliders.candidate_failure import FileRecord
from evidence_dial.guard_scope import facts, record, admit, V5_SOURCE
from evidence_dial.io import write_json


def case(tmp_path,text,reason,ids):
    output=tmp_path/'output.txt';output.write_text(text)
    probes=tmp_path/'probes.json'
    write_json(probes,{'probes':[{'output':text,'native':{'output_ids':ids}}]})
    binding={'output_source':asdict(FileRecord.from_path(output)),
        'actual_instrument_source':{'execution_adapter_source':{'sha256':V5_SOURCE}},
        'native_contract':{'allowed_output':'Any abstention label; normal token-budget truncation is permitted'}}
    write_json(tmp_path/'candidate-failed.json',{'status':'failed','checkpoints':{'x':{
        'observations':[{'status':'fail','binding':binding,'reason':reason}]}}})
    q=tmp_path/'qualification.json'
    write_json(q,{'qualified':True,'accepted':True,'covered_media':['text'],
                  'actual_instrument_source':{'instrument_source':{'sha256':'fixture'}}})
    return probes,q


def test_legal_label_audit_preserves_failed_bytes_and_is_not_a_pass(tmp_path):
    probes,q=case(tmp_path,'UNKNOWN','UNKNOWN',[1,3])
    original=(tmp_path/'candidate-failed.json').read_bytes()
    record(tmp_path,probes,q)
    assert (tmp_path/'candidate-failed.json').read_bytes()==original
    admit(tmp_path,observe_saved=True)
    # A label-scope audit alone cannot admit training or promotion.
    with pytest.raises((ValueError,FileNotFoundError,KeyError)):
        admit(tmp_path)


def test_actual_repetition_cannot_receive_scope_override(tmp_path):
    probes,q=case(tmp_path,'CONF\nCONF\nCONF','Pervasive unrequested repetition',[1,2,3])
    with pytest.raises(ValueError,match='no proved'):
        record(tmp_path,probes,q)


def test_modified_failure_cannot_reuse_prior_audit(tmp_path):
    probes,q=case(tmp_path,'UNKNOWN','UNKNOWN',[1,3])
    record(tmp_path,probes,q)
    path=tmp_path/'candidate-failed.json';path.write_text(path.read_text()+' ')
    with pytest.raises(ValueError,match='differ'):
        admit(tmp_path,observe_saved=True)


def test_replacement_catastrophe_stays_permanently_blocked(tmp_path):
    probes,q=case(tmp_path,'UNKNOWN','UNKNOWN',[1,3])
    record(tmp_path,probes,q)
    write_json(tmp_path/'candidate-confirmed-failed-native-text.json',{'status':'failed'})
    with pytest.raises(ValueError,match='cannot be overridden'):
        admit(tmp_path,observe_saved=True)
