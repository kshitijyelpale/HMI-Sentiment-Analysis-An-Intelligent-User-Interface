import React, { useState } from 'react';

import { Card, Container, Table } from 'react-bootstrap';
import axios from 'axios';

const ReviewRating = () => {
  const [rating, setRating] = useState(0.5);

  const review = axios.get('http://localhost:8080/review');
  axios.post('http://localhost:8080/cdcd', {
    rating: rating,
  });

  return (
    <Container>
      <Card>
        <Card.Body>
          <Card.Text>This is a movie review.</Card.Text>
          <div>
            <textarea readOnly>{review}</textarea>
          </div>
        </Card.Body>
        <Card.Body>
          <Card.Text id='rating'>
            Please provide rating for the above review.
          </Card.Text>
          <div className='my-5'>
            <label htmlFor='customRange1'>Example range</label>
            <input
              type='range'
              className='custom-range slider'
              id='rating'
              min={0.0}
              max={1.0}
              step={0.01}
              value={rating}
              onChange={(e) => setRating(e.target.value)}
              defaultValue={0.5}
            />
          </div>
          <h1>{rating}</h1>
        </Card.Body>
      </Card>
      <Table bordered hover>
        <thead>
          <tr>
            <th>#</th>
            <th>First Name</th>
            <th>Last Name</th>
            <th>Username</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1</td>
            <td>Mark</td>
            <td>Otto</td>
            <td>@mdo</td>
          </tr>
          <tr>
            <td>2</td>
            <td>Jacob</td>
            <td>Thornton</td>
            <td>@fat</td>
          </tr>
          <tr>
            <td>3</td>
            <td colSpan='2'>Larry the Bird</td>
            <td>@twitter</td>
          </tr>
        </tbody>
      </Table>
    </Container>
  );
};

export default ReviewRating;
