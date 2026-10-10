Native paper qualification remains outstanding. The fixture policy uses 3-second
soft acknowledgment, 12-second hard acknowledgment, 15-second cancellation,
5-second snapshot and 3-second protection deadlines solely to exercise boundaries.
These numbers are not measured live or paper acceptance behavior.

An operator-authorized paper session must qualify the exact frozen source hash
on a paper account, recording native callback times and current order echoes,
execution IDs, complete read boundaries, positions, source/client/session identity,
and request receipts. It must cover delayed acceptance; execDetails before status;
cancel refusal plus synthetic SDK Cancelled; true native cancellation and later
execution closure; late and partial entry fills; protective revisions and OCA
reductions; stop/time exit races; journal failure/redelivery; disconnect/restart
without resend; unknown owner exposure blocking the peer; independent peer-market
waiting; and atomic account risk reservations. Native protective reduction and
automatic close behavior need separate design/qualification before automation.

Required installer case_result keys are acceptance, cancel_race,
late_partial_fill, protective_revision, restart_no_resend, disconnect_no_resend,
owner_unknown_block and risk_atomicity. All must be PASS in a frozen
paper-native-lifecycle evidence JSON with mode paper, a DU paper account,
operator_approved true, exact source_sha256, observed_ack_max_seconds and
observed_protection_ack_max_seconds. The approved paper_qualified policy must
reference that evidence hash and source hash. Hard acknowledgment must exceed the
separately measured entry maximum; protection must exceed its own measured maximum.
Finite bounded deadlines and explicit approval are enforced. Handwritten fixture
assertions test parsing only and are not sufficient operational evidence.

Manual reconciliation remains mandatory for unproven history, corrections,
unexpected native body/quantity changes, a late entry while old exit closure is
uncertain, or an exit that reverses a flat position. Automatic emergency flatten
and coordination close remain disabled. No new qualification run is authorized
by publishing this offline package.
