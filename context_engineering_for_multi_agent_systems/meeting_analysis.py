from commons.utils import initialize_clients

client, _ = initialize_clients()

def call_llm(prompt):
    return client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "使用中文回答"},{"role": "user", "content": prompt}]
    )

meeting_transcript = """
 Tom: Morning all. Coffee is still kicking in.
 Sarah: Morning, Tom. Right, let's jump in. Project Phoenix 
timeline. Tom, you said the backend components are on track?
 Tom: Mostly. We hit a small snag with the payment gateway 
integration. It's... more complex than the docs suggested. We might need 
another three days.
 Maria: Three days? Tom, that's going to push the final testing 
phase right up against the launch deadline. We don't have that buffer.
 Sarah: I agree with Maria. What's the alternative, Tom?
 Tom: I suppose I could work over the weekend to catch up. I'd 
rather not, but I can see the bind we're in.
 Sarah: Appreciate that, Tom. Let's tentatively agree on that.
Maria, what about the front-end?
 Maria: We're good. In fact, we're a bit ahead. We have some extra 
bandwidth.
 Sarah: Excellent. Okay, one last thing. The marketing team wants to 
do a big social media push on launch day. Thoughts?
 Tom: Seems standard.
 Maria: I think that's a mistake. A big push on day one will swamp 
our servers if there are any initial bugs. We should do a soft launch, 
invite-only for the first week, and then do the big push. More controlled.
 Sarah: That's a very good point, Maria. A much safer strategy.
Let's go with that. Okay, great meeting. I'll send out a summary.
 Tom: Sounds good. Now, more coffee.
 """

prompt_g2 = f"""
 Analyze the following meeting transcript. Your task is to isolate 
the substantive content from the conversational noise.
 - Substantive content includes: decisions made, project updates, 
problems raised, and strategic suggestions.
 - Noise includes: greetings, pleasantries, and off-topic remarks 
(like coffee).
 Return ONLY the substantive content.
 Transcript:
 ---
{meeting_transcript}
 ---
 """
response_g2 = call_llm(prompt_g2)
substantive_content = response_g2.choices[0].message.content
print("--- SUBSTANTIVE CONTENT ---")
print(substantive_content)

previous_summary = """
In our last meeting, we finalized the goals for Project Phoenix and
assigned
backend
work
to
Tom and front - end
to
Maria.
"""
prompt_g3 = f"""
Context: The summary of our last meeting was: "{previous_summary}"
Task: Analyze the following substantive content from our new meeting.
Identify and summarize ONLY the new developments, problems, or decisions 
that have occurred since the last meeting.
New Meeting Content:
---
{substantive_content}
---
"""
new_developments = None
try:
    response_g3 = call_llm(prompt_g3)
    new_developments = response_g3.choices[0].message.content
    print("--- NEW DEVELOPMENTS SINCE LAST MEETING ---")
    print(new_developments)
except Exception as e:
    print(f"An error occurred: {e}")

prompt_g4 = f"""Task: Analyze the following meeting content for implicit social dynamics 
and unstated feelings. Go beyond the literal words.
- Did anyone seem hesitant or reluctant despite agreeing to something?
- Were there any underlying disagreements or tensions?
- What was the overall mood?
Meeting Content:
---
{substantive_content}
---
"""

try:
    response_g4 = call_llm(prompt_g4)
    implicit_threads = response_g4.choices[0].message.content
    print("--- IMPLICIT THREADS AND DYNAMICS ---")
    print(implicit_threads)
except Exception as e:
    print(f"An error occurred: {e}")

prompt_g5 = f"""
Context: In the meeting, Maria suggested a 'soft launch' to avoid server 
strain, and also mentioned her team has 'extra bandwidth'.
Tom is facing a 3-day delay on the backend.
Task: Propose a novel, actionable idea that uses Maria's team's extra 
bandwidth to help mitigate Tom's 3-day delay. Combine these two separate 
pieces of information into a single solution.
"""
try:
    response_g5 = call_llm(prompt_g5)
    novel_solution = response_g5.choices[0].message.content
    print("--- NOVEL SOLUTION PROPOSED BY AI ---")
    print(novel_solution)
except Exception as e:
    print(f"An error occurred: {e}")

prompt_g6 = f"""
Task: Create a final, concise summary of the meeting in a markdown table.
Use the following information to construct the table.
- New Developments: {new_developments}
The table should have three columns: "Topic", "Decision/Outcome", and "Owner".
"""
final_summary_table = None
try:
    response_g6 = call_llm(prompt_g6)
    final_summary_table = response_g6.choices[0].message.content
    print("--- FINAL MEETING SUMMARY TABLE ---")
    print(final_summary_table)
except Exception as e:
    print(f"An error occurred: {e}")

prompt_g7 = f"""
Task: Based on the following summary table, draft a polite and professional 
follow-up email to the team (Sarah, Tom, Maria).
The email should clearly state the decisions made and the action items for each 
person.
Summary Table:
---
{final_summary_table}
---
"""

try:
    response_g7 = call_llm(prompt_g7)
    follow_up_email = response_g7.choices[0].message.content
    print("--- DRAFT FOLLOW-UP EMAIL ---")
    print(follow_up_email)
except Exception as e:
    print(f"An error occurred: {e}")
