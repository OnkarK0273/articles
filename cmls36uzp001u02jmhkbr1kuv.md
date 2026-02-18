---
title: "Session Based Auth System [state-full]"
seoTitle: "Session Based Authentication System"
seoDescription: "Explore the workings of session-based authentication, its implementation in monoliths, and methods to handle multi-server setups effectively"
datePublished: Wed Feb 18 2026 13:47:31 GMT+0000 (Coordinated Universal Time)
cuid: cmls36uzp001u02jmhkbr1kuv
slug: session-based-auth-system-state-full
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1771419715447/10c15e23-0c48-4c8a-a19d-d093a0ada280.png
tags: authentication, session-based-authentication

---

* Session based auth is traditional method of auth and it is `state-full` means we store `sessions` of user in memory or DB and share `session_id` to user to verity his identity
    
* This method is gold standard for `monolith` application where front-end and back-end are served from same domain.
    

# How it’s work

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1771420012080/c66a1adf-94c6-4103-a0a9-5975c0b5b091.png align="center")

1. **Login** - user submit their credential to server
    
2. **Verification and creation** - if credential is correct against DB, it create a `session` in DB and generate a unique `session_id`
    
3. **Cookie** - Server send back response with header of `set_cookie`includes `sesson_id`
    
4. **Browser** - it automatically store the `cookie`
    
5. **Subsequent request** - in every future server request, browser attach cookie of `session_id` in header requests.
    
6. **Server validation** - it pull the `session_id` from header cookie and look up in session storage if it matches then `process` the request.
    

# Methods in session based auth

I said it earlier it is not idle for multi-server system, but there is practical methods to achieve this using following methods.

## 1\. Replication of Session storage

### Why we use this method ?

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1771420324905/953c2cc5-e527-4b75-8119-d611b1c98376.png align="center")

* Incase of multi server user 1st login req. go to `auth_server` that create and save session in `session_storage_DB-1`
    
* Because of `load-balancer` the next req. with `session_id` header is route to `server-2` but it failed to validation because their session is not stored in `session_storage_DB-2` it leads logout of user.
    
* To resolve this issue we use replication session storage
    

### How it’s works ?

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1771420585639/e871421b-9ebf-4ea1-b54d-471c618a3983.png align="center")

When any one server creates or updates a session, it immediately broadcasts that change to the data-grid, which then updates all other servers.

### Drawback

* Memory overhead - each server contain all the session which consumes your all available RAM
    
* Complexity - To keep server session up to date in Realtime is difficult
    

## 2\. Centralized Session storage

### why we use this method

* It resolves memory overhead issue
    
* To make system less complex
    

### How it’s work?

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1771420978975/35ae80b8-6cac-4453-93d2-f0bd920285f0.png align="center")

Here we use common `session_storage_DB` for both servers can fetch and validate the `session_id` against single `session_storage_DB`

### Drawback

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1771421143765/96afb0d9-8981-47a0-9a5b-27e50b3b076b.png align="center")

**Single Point of Failure:** If the `Session_storage_DB` goes down, no one can log in