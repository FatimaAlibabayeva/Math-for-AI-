Movie Recommendation System (Mini-Project)
**Overview**

This is a simple Movie Recommendation System built using Python and Linear Algebra concepts. The system recommends movies to a user based on the preferences of the most similar users. It demonstrates how dot product and cosine similarity can be used in AI recommendation systems.

**Features**

Represents user preferences as vectors (ratings of movies).

Computes similarity between users using dot product or cosine similarity.

Recommends movies that the most similar user liked but the target user has not watched yet.

Simple, easy-to-understand, and extendable for larger datasets.

**How It Works**

User Ratings:
Each user is represented by a vector of ratings.
Example:

user1 = [5, 4, 0, 1]  # 0 = not watched
user2 = [4, 5, 0, 1]
user3 = [1, 0, 5, 4]


Similarity Calculation:
Compute similarity between users using cosine similarity:

similarity(u, v) = dot(u, v) / (||u|| * ||v||)


Higher similarity → users have more similar tastes.

Find Most Similar User:

The system finds the user with the highest similarity to the target user.

**Recommendation:**

Recommend movies that:

The most similar user has liked (rating > 0)

The target user has not watched yet (rating = 0)

**Future Improvements**

Use larger datasets with real movie names.

Implement weighted ratings or implicit feedback for better recommendations.

Visualize users and movies in 2D/3D space to see similarity.

Extend to multiple recommendations and ranking.

**Learnings**

How dot product and cosine similarity measure vector similarity.

How linear algebra concepts apply in AI recommendation systems.

Basics of building a simple recommendation engine in Python.
