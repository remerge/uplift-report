# Remerge uplift tracking group assignment algorithm

# DEPENDENCIES:
# The group assignment has one external dependency "xxhash", which is
# a hashing algorithm we use for quality and performance reasons. To install:
#     pip3 install xxhash==2.0.0
#
# It successfully completes the SMHasher test suite which evaluates collision,
# dispersion and randomness qualities of hash functions. To learn more about
# the quality of xxHash32 check the project repository:
# https://github.com/Cyan4973/xxHash

# USAGE:
# To run you need to provide the following arguments:
#   --uuid            raw user IDFA or AAID in UUID string format
#   --control-share   control group share in between 0 and 1
#   --reshuffle-date  reshuffle date of the test in YYYY-MM-DD format
#   --salt            salt for the group assignment hash [DEPRECATED]
#
# Only one of the "reshuffle_date" or "salt" arguments is required.
# A random salt was used in an older implementation of our uplift tracking.
# The current version only uses the test reshuffle dates as salt.
# The parameter applicable to your uplift test will be supplied by Remerge.
#
# Example:
#     python3 remerge_uplift_group_assignment.py \
#             --control-share 0.2 \
#             --reshuffle-date 2020-10-25 \
#             --uuid a6171e63-0d24-4323-86e0-2d1367dd133a

import argparse
from datetime import datetime, timezone
from hashlib import sha1
from xxhash import xxh32

# fixed seed we always use in our application
XXHASH32_SEED = 6530

GROUP_TEST = 'test'
GROUP_CONTROL = 'control'


def main():
    parser = argparse.ArgumentParser(
        description='Remerge uplift test control group assignment')
    parser.add_argument('--uuid', type=str, required=True,
                        help='raw user IDFA or AAID in UUID string format')
    parser.add_argument('--control-share', type=float, required=True,
                        help='control group share in between 0 and 1')
    parser.add_argument('--reshuffle-date',
                        type=lambda s: datetime.strptime(s, '%Y-%m-%d'),
                        help='reshuffle date of the test in YYYY-MM-DD format')
    parser.add_argument('--salt', dest='salt', type=int,
                        help='salt for the group assignment hash [deprecated]')

    args = parser.parse_args()

    group, hash = assign_user_to_group(args.uuid, args.control_share,
                                       reshuffle_date=args.reshuffle_date,
                                       salt=args.salt)
    print(group, hash)


def assign_user_to_group(uuid: str,
                         control_share: float,
                         salt: int = None,
                         reshuffle_date: datetime = None) -> (str, float):
    """Returns the group name and float hash value."""

    salt = pick_salt(reshuffle_date, salt)
    hash = group_assignment_hash(uuid, salt)
    if hash <= control_share:
        return GROUP_CONTROL, hash
    return GROUP_TEST, hash


def pick_salt(reshuffle_date: datetime, salt: int) -> int:
    """Calculates salt from reshuffle_date if provided."""

    if salt is None and reshuffle_date is None:
        raise ValueError('reshuffle_date or salt is required')

    if salt is not None and reshuffle_date is not None:
        raise ValueError('salt and reshuffle_date provided, pass just one')

    if reshuffle_date is not None:
        # use reshuffle_date as UNIX nanosecond timestamp as salt
        seconds = reshuffle_date.replace(tzinfo=timezone.utc).timestamp()
        salt = int(seconds*1e9)

    return salt


def group_assignment_hash(uuid: str, salt: int) -> float:
    """Calculates the xxHash32 of the user ID and salt."""

    hashed_user_id = sha1(uuid.encode('utf-8')).digest()
    salt_bytes = salt.to_bytes(8, byteorder='little')

    hash = xxh32(seed=XXHASH32_SEED)
    hash.update(hashed_user_id)
    hash.update(salt_bytes)

    # scale 32 bit unsigned integer range to [0.0 - 1.0] float
    # through division by the maximum value
    max_uint32 = (1 << 32)-1
    return hash.intdigest() / max_uint32


main()
