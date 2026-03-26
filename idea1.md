# EPLB / fgate ideas

## Background

In the current implementation, `fgate` / `fgate-v2` only writes a predicted next-layer expert load into the EPLB statistics window, and real EPLB actions still happen only at a fixed `STEP_INTERVAL`.

This creates a core mismatch:

- **Prediction horizon is short**: only next-layer / near-future load is predicted.
- **EPLB action is slow**: real rearrangement / migration / remapping only happens at interval boundaries.

Because of this, fgate can easily degenerate into "just another number in the window" instead of a mechanism that changes short-term behavior.

---

## Better direction: two-level control

### Level 1: fast layer (cheap action)

Goal: **do not move weights and do not do global rearrangement; instead, use very cheap actions to reduce short-term hotspots and queue buildup.**

The key idea is:

> Convert fgate short-term predictions into an immediate dispatch / scheduling policy on top of the resources that already exist.

### What counts as a cheap action?

A fast-layer action should satisfy most of the following:

- no expert weight movement
- no global logical <-> physical remapping
- no heavy cross-rank synchronization
- can be refreshed frequently (for example every 1~8 decode steps)
- only changes lightweight control metadata such as dispatch policy, quota, scheduling order, or buffer reservation

So this layer is about **control**, not **migration**.

---

## Best fast-layer use case: predictive dispatch among existing replicas

### Assumption

A logical expert already has multiple physical replicas.

For example, logical expert `E17` may already have 3 physical replicas:

- `P17_a`
- `P17_b`
- `P17_c`

Because these replicas share the same weights, sending a token to one replica or another usually does **not** change model semantics.

So the router can continue to choose the **logical expert**, while the fast layer decides:

> Once a token has already been routed to logical expert `E17`, which physical replica should handle it?

---

## How the fast layer uses fgate

Maintain a short-horizon predicted demand for each logical expert:

- `pred[e]`: predicted short-term demand for logical expert `e`

Also maintain short-term runtime signals for each physical replica:

- `inflight[p]`: current unfinished token count
- `service_rate[p]`: recent service throughput
- `comm_cost[p]`: communication cost to this replica
- `device_load[p]`: current load on the device that hosts this replica

For the replica set `R(e)`, compute a lightweight dispatch score such as:

```text
score(p|e) = alpha * inflight[p]
           + beta  * pred[e] / |R(e)|
           + gamma * comm_cost[p]
           + delta * device_load[p]
```

When a token is already assigned to logical expert `e`:

- prefer the replica with lower `score`
- or distribute traffic among replicas using a softmax / quota-based policy

This way, the fgate prediction is no longer just a statistic. It immediately changes actual short-term dispatch.

---

## More practical fast-layer version: short-term quota / reservation

Instead of only scoring replicas on the fly, the system can also assign a short-lived quota to the replicas of a predicted-hot expert.

Example: if fgate predicts that `E17` will get hotter over the next 8 decode ticks, assign a temporary quota such as:

- `P17_a`: 40%
- `P17_b`: 35%
- `P17_c`: 25%

Then, for the next small window:

- route traffic according to the quota first
- if one replica uses up its quota, spill to backup replicas
- expire the quota automatically after the short window ends

The idea is:

> Reserve short-term capacity before the traffic surge arrives, instead of reacting only after queues have already formed.

---

## If there are no redundant experts, what can the fast layer still do?

If a logical expert has only one physical replica, replica-level balancing is impossible. But there are still useful cheap actions:

### A. Request / microbatch staggering

If multiple requests are likely to hit the same hot expert, avoid placing them into the same decode microbatch when possible.

Useful techniques:

- request grouping
- microbatch reordering
- decode staggering

This does not change model semantics; it only changes short-term scheduling.

### B. Communication and buffer reservation

If a communication path is likely to get hot soon, prepare early:

- pre-reserve token buffers
- pre-allocate receive space
- prepare all-to-all metadata in advance

This is weaker than replica dispatch, but it is still a real cheap action.

---

## Why is this better than waiting for real load?

Because real load is often observed only after queues already exist.

In other words:

> If you only react to real load, congestion is already happening.

The fast layer makes fgate useful by doing something earlier:

> Use short-horizon predictions to spread traffic before the queue actually builds up.

So the real value is not just "more statistics". The value comes from:

- short-horizon prediction
- a control action on the same short horizon
- immediate effect on dispatch / quota / scheduling

---

## Relationship with slow EPLB

### Fast layer (cheap action)

Handles:

- short-term congestion in the next tens of steps
- queue spikes on a single replica
- microbatch-level imbalance
- short-term communication hotspots

### Slow layer (real EPLB rearrangement)

Handles:

- long-term hot experts
- which experts should be replicated
- where replicas should be placed
- whether full migration / rearrangement is worth doing

In one sentence:

- fast layer: **hold the line now**
- slow EPLB: **fix the layout later**

---

## Design principle

The most important rule for the fast layer is:

> **Do not change logical routing if possible; only change physical replica dispatch and short-term scheduling.**

Reason:

- changing logical routing may change model behavior
- changing replica choice is usually a systems-level optimization and preserves semantics

So the most natural design is:

1. the router still selects the logical expert
2. fgate predicts short-term logical-expert demand
3. the system translates this into short-term replica dispatch / quota / scheduling policies
4. slow EPLB still handles replication, migration, and global rearrangement

---

## Core summary

To make fgate genuinely useful for EPLB, it is not enough to write predictions into a window. The prediction must drive a control action with a matching time scale.

The role of the fast layer is exactly this:

> Turn fgate short-term predictions into short-term dispatch decisions on top of the existing replicas and resources.

Without this layer, fgate can collapse into "just another statistic".
With this layer, fgate gets a real chance to reduce short-term hotspots before slow EPLB takes over.
