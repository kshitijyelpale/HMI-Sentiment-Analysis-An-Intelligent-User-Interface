import React from 'react';
import { Button, Col, Container, Form, Row } from 'react-bootstrap';

const LoginPage = () => {
  return (
    <>
      <Container>
        <h1 className='shadow-sm text-info mt-5 p-3 text-center rounded'>
          Login
        </h1>
        <Row className='mt-5'>
          <Col
            lg={5}
            md={6}
            sm={12}
            className='p-5 m-auto shadow-sm rounded-lg'
          >
            <Form>
              <Form.Group controlId='email'>
                <Form.Label>Email address</Form.Label>
                <Form.Control type='email' placeholder='Enter email' />
              </Form.Group>

              <Form.Group controlId='pwd'>
                <Form.Label>Password</Form.Label>
                <Form.Control type='password' placeholder='Password' />
              </Form.Group>

              <Button variant='primary btn-block mt-3' type='submit'>
                Login
              </Button>
            </Form>
          </Col>
        </Row>
      </Container>
    </>
  );
};

export default LoginPage;
