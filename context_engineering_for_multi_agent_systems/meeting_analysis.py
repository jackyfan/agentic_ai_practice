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
